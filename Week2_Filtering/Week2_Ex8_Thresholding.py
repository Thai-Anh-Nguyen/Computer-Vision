import cv2
class ThresholdingProcessor:
    def __init__(self):
        pass
    def convert_to_threshold(self, bgr_img):
        if bgr_img is None:
            return None

        #convert BMP to Threshold
        threshold = cv2.threshold(bgr_img, 128, 255, cv2.THRESH_BINARY)[1]
        return threshold