import cv2
import numpy as np

class DilationProcessor:
    def __init__(self):
        pass
    def convert_to_dilation(self, bgr_img):
        if bgr_img is None:
            return None
        
        #convert BMP to Dilation
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(bgr_img, kernel, iterations=1)
        return dilation