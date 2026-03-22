# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread( r'D:\Computer Vision\Lessons\Week 6 Edge Detection\test.jpg',cv2.IMREAD_GRAYSCALE)

# # Canny edge detection, double threshold: 100 and 200
# edges = cv2.Canny(img, 150, 200)

# plt.imshow(edges, cmap='gray')
# plt.title('Sobel Edge Canny')
# plt.axis('off')
# plt.show()


import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Computer Vision\Lessons\Week 6 Edge Detection\test.jpg', cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (lo, hi), label in zip(axes,
    [(150, 200), (50, 150), (30, 100)],
    ['150/200 (yours)', '50/150', '30/100']):
    edges = cv2.Canny(img, lo, hi)
    ax.imshow(edges, cmap='gray')
    ax.set_title(label)
    ax.axis('off')

# plt.show()
print(img.shape)  # prints (height, width, channels)