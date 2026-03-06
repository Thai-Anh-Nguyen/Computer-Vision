import cv2

class LaplacianProcessor:
    def __init__(self):
        pass
    def convert_to_laplacian(self, bgr_img):
        if bgr_img is None:
            return None
        
        #convert BMP to Laplacian
        laplacian = cv2.Laplacian(bgr_img, cv2.CV_64F)
        return laplacian