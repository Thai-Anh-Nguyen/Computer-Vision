import cv2
import threading
import time


class VideoCamera:
    """
    Wraps an OpenCV VideoCapture in a background thread.

    Why a background thread?
    ------------------------
    cap.read() BLOCKS until a frame arrives from the camera — this can
    take 30-100 ms. If we called it directly inside a web route, the
    entire server would freeze waiting for each frame. Instead, one
    dedicated thread runs cap.read() in a tight loop and stores the
    latest frame. The web server just grabs that stored frame instantly
    without ever blocking.

    Supported sources (passed to start()):
        0, 1, 2 ...         → USB / built-in webcam index
        "http://x.x.x.x/video" → iPhone IP camera (EpocCam, Camo, etc.)
        "rtsp://..."        → RTSP network camera
    """

    def __init__(self):
        # The latest decoded frame as a BGR numpy array (H, W, 3).
        # None means no frame has arrived yet.
        self.frame = None

        # Lock protects self.frame so the reader thread and the web
        # server never read/write it at the exact same moment.
        self.lock = threading.Lock()

        # Internal state
        self._source = None       # camera URL / index currently in use
        self._running = False     # signals the reader thread to stop
        self._thread = None       # reference to the background thread
        self._cap = None          # the OpenCV VideoCapture object

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, source):
        """
        Open a new camera source and start the background reader.

        If this camera is already streaming the same source, do nothing.
        If it's streaming a different source, stop first then restart.

        Args:
            source: int or str — webcam index, HTTP URL, or RTSP URL
        """
        # Try to coerce numeric strings ("0", "1") to int so OpenCV
        # opens them as webcam indices rather than filenames.
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass  # keep as string (URL / device path)

        # Already streaming this exact source — nothing to do
        if self._running and self._source == source:
            return

        # Stop any existing stream before opening a new one
        self.stop()

        self._source = source
        self._running = True

        # daemon=True means this thread dies automatically when the
        # main program exits — no manual cleanup needed on shutdown.
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Signal the reader thread to stop and wait for it to exit.
        Releases the camera resource afterwards.
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)  # wait up to 2 s
            self._thread = None

        self._release_cap()
        self.frame = None   # clear stale frame so callers see None

    def get_frame_bgr(self):
        """
        Return a copy of the latest BGR frame, or None if not ready.

        We return a COPY so the caller can freely modify the array
        without racing against the reader thread overwriting it.
        """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_jpeg(self, quality: int = 80) -> bytes | None:
        """
        Return the latest frame encoded as JPEG bytes, or None.

        Used by the MJPEG streaming route in main.py.

        Args:
            quality: JPEG quality 0-100 (80 is a good default —
                     lower = smaller payload = smoother stream)
        """
        frame = self.get_frame_bgr()
        if frame is None:
            return None

        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    @property
    def is_running(self) -> bool:
        """True if the reader thread is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reader(self):
        """
        Background thread: continuously read frames from the camera.

        Flow:
            1. Open the capture device.
            2. Read frames in a loop, storing each one under self.lock.
            3. If the device disconnects, wait 2 s then try to reopen.
            4. Exit cleanly when self._running is set to False.
        """
        self._open_cap()

        while self._running:
            # --- Handle unopened / disconnected capture ---
            if self._cap is None or not self._cap.isOpened():
                time.sleep(2.0)
                self._open_cap()
                continue

            # --- Read one frame ---
            ok, frame = self._cap.read()

            if not ok or frame is None:
                # Camera returned an empty frame — could be a hiccup
                # or a real disconnect. Sleep briefly and retry.
                time.sleep(0.05)
                continue

            # --- Store frame (protected by lock) ---
            with self.lock:
                self.frame = frame  # reader owns this write

            # Small sleep to avoid burning 100 % CPU.
            # 0.02 s → ~50 fps ceiling, more than enough.
            time.sleep(0.02)

        # Thread is exiting — release the camera
        self._release_cap()

    def _open_cap(self):
        """Try to open self._source with OpenCV."""
        self._release_cap()
        try:
            if isinstance(self._source, int):
                # Webcam index → DirectShow (Windows native, best for local cams)
                # CAP_FFMPEG on a webcam index causes grey screen on Windows
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            else:
                # URL / RTSP / HTTP → FFMPEG handles network streams best
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        except Exception:
            # Last resort — let OpenCV pick the backend itself
            self._cap = cv2.VideoCapture(self._source)

    def _release_cap(self):
        """Release VideoCapture if it exists."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None