import time
import numpy as np
import Lessons

# ---------------------------------------------------------------------------
# Safe imports
# ---------------------------------------------------------------------------

try:
    from Lessons.Week1_Capturering.Week1_captureSaveImg import CaptureSaveImgProcessor
except ImportError:
    CaptureSaveImgProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor
except ImportError:
    GrayscaleProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex2_Gaussian import GaussianProcessor
except ImportError:
    GaussianProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex3_Median import MedianBurProcessor
except ImportError:
    MedianBurProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex4_Sobel import SobelProcessor
except ImportError:
    SobelProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex5_Laplacian import LaplacianProcessor
except ImportError:
    LaplacianProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex6_Sharpening import SharpeningProcessor
except ImportError:
    SharpeningProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex7_Bilateral import BilateralProcessor
except ImportError:
    BilateralProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex8_Thresholding import ThresholdingProcessor
except ImportError:
    ThresholdingProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex9_Erosion import ErosionProcessor
except ImportError:
    ErosionProcessor = None

try:
    from Lessons.Week2_Filtering.Week2_Ex10_Dilation import DilationProcessor
except ImportError:
    DilationProcessor = None


def _init(cls):
    return cls() if cls is not None else None


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------

class ImageProcessor:

    def __init__(self):
        self.camera_matrix     = None
        self.dist_coeffs       = None
        self.homography_matrix = None
        self.previous_frame    = None
        self.tracked_objects   = []

        self.save_img          = _init(CaptureSaveImgProcessor)
        self.filter_gray       = _init(GrayscaleProcessor)
        self.filter_gaussian   = _init(GaussianProcessor)
        self.filter_median     = _init(MedianBurProcessor)
        self.filter_sobel      = _init(SobelProcessor)
        self.filter_laplacian  = _init(LaplacianProcessor)
        self.filter_sharpen    = _init(SharpeningProcessor)
        self.filter_bilateral  = _init(BilateralProcessor)
        self.filter_threshold  = _init(ThresholdingProcessor)
        self.filter_erosion    = _init(ErosionProcessor)
        self.filter_dilation   = _init(DilationProcessor)

        # -------------------------------------------------------------------
        # STEP REGISTRY
        # -------------------------------------------------------------------
        # A dict that maps a filter name (string the frontend sends)
        # to the actual function that applies it.
        #
        # Why a dict instead of if/elif?
        #   Clean O(1) lookup. Adding a new filter = one new line.
        #   Nothing else in the codebase needs to change.
        #
        # Why lambda?
        #   Each filter has a different method name. Lambda wraps them
        #   into one uniform interface: takes a frame, returns a frame.
        #
        # _register() skips the entry if the import failed (instance=None)
        # so broken week folders never crash the app.
        # -------------------------------------------------------------------
        self._steps: dict = {}

        self._register("grayscale",  self.filter_gray,      lambda f: self.filter_gray.convert_to_grayscale(f))
        self._register("gaussian",   self.filter_gaussian,  lambda f: self.filter_gaussian.convert_to_gaussian(f))
        self._register("median",     self.filter_median,    lambda f: self.filter_median.convert_to_median(f))
        self._register("sobel",      self.filter_sobel,     lambda f: self.filter_sobel.convert_to_sobel(f))
        self._register("laplacian",  self.filter_laplacian, lambda f: self.filter_laplacian.convert_to_laplacian(f))
        self._register("sharpening", self.filter_sharpen,   lambda f: self.filter_sharpen.convert_to_sharpening(f))
        self._register("bilateral",  self.filter_bilateral, lambda f: self.filter_bilateral.convert_to_bilateral(f))
        self._register("threshold",  self.filter_threshold, lambda f: self.filter_threshold.convert_to_threshold(f))
        self._register("erosion",    self.filter_erosion,   lambda f: self.filter_erosion.convert_to_erosion(f))
        self._register("dilation",   self.filter_dilation,  lambda f: self.filter_dilation.convert_to_dilation(f))

    def _register(self, name: str, instance, fn):
        """Add filter to registry only if its import succeeded."""
        if instance is not None:
            self._steps[name] = fn

    def available_steps(self) -> list:
        """
        Return filter names that actually loaded successfully.
        Exposed via GET /filters in main.py so the frontend can
        build the dropdown from real availability, not a hardcoded list.
        """
        return list(self._steps.keys())

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------

    def process_frame(self, bgr_img: np.ndarray, step: str = "none"):
        """
        Apply the selected filter to the frame.

        Args:
            bgr_img : raw BGR frame from the camera
            step    : filter name sent from the frontend dropdown,
                      or "none" to return the original frame unchanged

        Returns:
            processed_img   : np.ndarray
            results         : dict  (reserved for future detections)
            process_time_ms : float
        """
        if bgr_img is None:
            raise ValueError("Input frame is None")

        start   = time.perf_counter()
        results = {}

        if step == "none" or step not in self._steps:
            processed_img = bgr_img.copy()
        else:
            processed_img = self._steps[step](bgr_img)

        process_time_ms = (time.perf_counter() - start) * 1000
        return processed_img, results, process_time_ms