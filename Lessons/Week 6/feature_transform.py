import cv2
import numpy as np

# img = cv2.imread(r'D:\Computer Vision\Lessons\Week 6\img3.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# img_kp = cv2.drawKeypoints(img, kp, None)
# cv2.imwrite('img_kp.jpg', img_kp)
# cv2.imshow('SIFT Keypoints', img_kp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # Demonstrates SIFT feature detection.
# # Slide 10: Feature Matching Example (Python/ OpenCV)
path1 = r'D:\Computer Vision\Lessons\Week 6\img1.jpg'
path2 = r'D:\Computer Vision\Lessons\Week 6\img2.jpg'
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# cv2.imwrite('img_matches.jpg', img_matches)
# cv2.imshow('Matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if len(good) > 10:
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches found.")
	
	
height, width = img2.shape
result = cv2.warpPerspective(img1, H, (width * 2, height))
result[0:height, 0:width] = img2
cv2.imwrite('panorama.jpg', result)
cv2.imshow('Panorama', result)
cv2.waitKey(0)
cv2.destroyAllWindows()