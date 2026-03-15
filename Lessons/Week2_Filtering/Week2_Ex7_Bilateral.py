import cv2

class BilateralProcessor:
    def __init__(self):
        pass

    def convert_to_bilateral(self, image):
        if image is None:
            return None
        
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        return bilateral