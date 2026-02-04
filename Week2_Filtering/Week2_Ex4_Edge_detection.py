import cv2

class EdgeDectectProcessor:
    def detect_edges(self, bgr_img):
        # TODO: Implement edge detection
        if bgr_img is None:
            return None

        # Convert BMP to Grayscale
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Canny edge detection, double threshold: 100 and 200
        edges = cv2.Canny(gray_img, 100, 200)
        return edges