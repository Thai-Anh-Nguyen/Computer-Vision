import cv2

class MedianBurProcessor:
    def __init__(self):
        pass

    def convert_to_median(self, bgr_img):
        if bgr_img is None:
            return None
        
        #convert BMP to Median
        median = cv2.medianBlur(bgr_img, 5)
        return median