import cv2
class GaussianProcessor:
    def __init__(self):
        pass
    
    # =============================================================================
    # STEP 1: BASIC IMAGE CAPTURE (Weeks 1-2)
    # Topic: Introduction to Computer Vision, Images as Functions & Filtering
    # =============================================================================
    
    def convert_to_gaussian(self, bgr_img):
        """
        Apply Gaussian filtering to reduce noise
        
        Args:
            img: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation
            
        Returns:
            Filtered image
        """
        # TODO: Implement Gaussian filtering
        # Hint: Use cv2.GaussianBlur

        if bgr_img is None:
            return None
        kernel_size=(15, 15)
        sigma=5.0
        filtered_img = cv2.GaussianBlur(bgr_img, kernel_size, sigma)
        return filtered_img
    