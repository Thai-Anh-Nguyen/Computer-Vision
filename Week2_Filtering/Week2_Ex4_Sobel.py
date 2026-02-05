import cv2

class SobelProcessor:
    def __init__(self):
        pass

    def convert_to_sobel(self, bgr_img):
        if bgr_img is None:
            return None
        
        #convert BMP to Sobel
        sobelx = cv2.Sobel(bgr_img, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(bgr_img, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        return sobel