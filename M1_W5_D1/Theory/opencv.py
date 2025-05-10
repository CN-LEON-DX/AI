import cv2

# read img
# imread("path_", 0 - gray image, 1 - color, -1)
data = cv2.imread('mango.png', 1)

# print(data)
# save image 

cv2.imwrite('processed_image.jpg', data)
