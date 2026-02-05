import time
import cv2
import numpy as np
import os
from Week1_Capturering.Week1_captureSaveImg import CaptureSaveImgProcessor
from Week2_Filtering.Week2_Ex1_Grayscale import GrayscaleProcessor
from Week2_Filtering.Week2_Ex2_Gaussian import GaussianProcessor
from Week2_Filtering.Week2_Ex3_Median import MedianBurProcessor
from Week2_Filtering.Week2_Ex4_Sobel import SobelProcessor
from Week2_Filtering.Week2_Ex5_Laplacian import LaplacianProcessor
from Week2_Filtering.Week2_Ex6_Sharpening import SharpeningProcessor
from Week2_Filtering.Week2_Ex7_Bilateral import BilateralProcessor
from Week2_Filtering.Week2_Ex8_Thresholding import ThresholdingProcessor
from Week2_Filtering.Week2_Ex9_Erosion import ErosionProcessor
from Week2_Filtering.Week2_Ex10_Dilation import DilationProcessor



"""
rtsp://admin:ACLAB2023@192.168.8.105:554/Streaming/channels/101
"""
class ImageProcessor:
    """
    Class for processing images from camera feed
    Implements computer vision techniques from ProjectProgress.txt
    """
    
    def __init__(self):
        """Initialize image processor with calibration parameters"""
        self.camera_matrix = None  # Camera calibration matrix
        self.dist_coeffs = None    # Distortion coefficients
        self.homography_matrix = None  # Homography transformation matrix
        self.previous_frame = None  # For motion detection
        self.tracked_objects = []   # For object tracking
        

        #Initialize all Processors
        self.filter_greyscale = GrayscaleProcessor()
        self.filter_gaussian = GaussianProcessor()
        self.filter_median = MedianBurProcessor()
        self.filter_sobel = SobelProcessor()
        self.filter_laplacian = LaplacianProcessor()
        self.filter_sharpening = SharpeningProcessor()
        self.filter_bilateral = BilateralProcessor()
        self.filter_thresholding = ThresholdingProcessor()
        self.filter_erosion = ErosionProcessor()
        self.filter_dilation = DilationProcessor()

    def process_frame(self, bgr_img):

        if bgr_img is None:
            raise ValueError("Input frame is None")
        
        start_time = time.perf_counter()
        results = {}
        
        ###################### WRITE YOUR PROCESS PIPELINE HERE #########################
        saveImg = CaptureSaveImgProcessor()
        
        # saved_unprocessed_img = saveImg.capture_and_save_image(bgr_img, "test_capture.bmp")
        # processed_img = self.filter_gaussian.convert_to_gaussian(bgr_img)
        # processed_img = self.filter_greyscale.convert_to_grayscale(bgr_img)
        # processed_img = self.filter_median.convert_to_median(bgr_img)
        # processed_img = self.filter_sobel.convert_to_sobel(bgr_img)
        # processed_img = self.filter_laplacian.convert_to_laplacian(bgr_img)
        # processed_img = self.filter_sharpening.convert_to_sharpening(bgr_img)
        # processed_img = self.filter_bilateral.convert_to_bilateral(bgr_img)
        # processed_img = self.filter_thresholding.convert_to_threshold(bgr_img)
        # processed_img = self.filter_erosion.convert_to_erosion(bgr_img)
        processed_img = self.filter_dilation.convert_to_dilation(bgr_img)
        # saved_processed_img = saveImg.capture_and_save_image(processed_img, "processed_capture.bmp")
        #################################################################################
    

        process_time_ms = (time.perf_counter() - start_time) * 1000
        
        return processed_img, results, process_time_ms
    
    def visualize_results(self, bgr_img, results):
        """
        Visualize all processing results on image
        
        Args:
            bgr_img: Original image
            results: Dictionary of results from process_frame
            
        Returns:
            Annotated image
        """

        pass
