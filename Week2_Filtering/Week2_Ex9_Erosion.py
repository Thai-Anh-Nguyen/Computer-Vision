import cv2
import numpy as np

class ErosionProcessor:
    def __init__(self):
        pass
    def convert_to_erosion(self, bgr_img):
        if bgr_img is None:
            return None
        
        #convert BMP to Erosion
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(bgr_img, kernel, iterations=1)
        return erosion