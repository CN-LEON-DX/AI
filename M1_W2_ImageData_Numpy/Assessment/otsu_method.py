import numpy as np
import cv2
import os

def computeDifference(bg_img, input_img):
    diff = cv2.absdiff(bg_img, input_img)
    # Chuyển sang ảnh grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return gray_diff

def computeBinaryMask(diff_single_channel):
    # Áp dụng Otsu's method để tự động tìm threshold
    _, binary_mask = cv2.threshold(diff_single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

def replaceBackground(bg_image1, bg_image2, ob_image):
    diff = computeDifference(bg_image1, ob_image)
    
    mask = computeBinaryMask(diff)
    
    output = np.where(mask == 255, ob_image, bg_image2)
    return output

if __name__ == "__main__":
    required_files = ['background1.png', 'background2.png', 'object.png']
    if not all(os.path.exists(f) for f in required_files):
        print("Error: Missing input files!")
        exit()
    bg1 = cv2.imread('background1.png')
    bg2 = cv2.imread('background2.png')
    obj = cv2.imread('object.png')
    target_size = (640, 480)
    bg1 = cv2.resize(bg1, target_size)
    bg2 = cv2.resize(bg2, target_size)
    obj = cv2.resize(obj, target_size)

    result = replaceBackground(bg1, bg2, obj)
    
    cv2.imwrite('output_otsu.png', result)
    print("'Output saved as ' output_otsu.png'")