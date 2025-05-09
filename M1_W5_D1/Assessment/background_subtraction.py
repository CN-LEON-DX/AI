# Background subtraction là một phương pháp lấy cảm hứng từ các phép tính đơn giản trên ma trận
# để thay đổi nền của đối tượng trong hình ảnh ?


# + Hệ thống giám sát an ninh, phân tích video 
# + Ứng dụng nghệ thuật 4d 

# Quy trình gồm 3 bc chính:

# + Tính toán sự khác biệt giữa nền và đối tượng 
# + Tạo mask nhị phân để phân biệt
# + Cuối cùng là thay thế bằng hình ảnh mới  


# + Công cụ thực hiện:
 
#  # OPENCV LIBRARY + NUMPY 

# + Kết quả cuối cùng là hình ảnh với đối tượng được giữ nguyên trên nền mới.


# Input: 2 Images: background and object in background 
# in the same background 

# Output: image with object inside the same background 

# Step to caculate the background substraction

# A = |I - B|,
# M = 
# {
#     0 If A < threshold
#     1 otherwise
# }

# O =
# {
#     F If M = 0
#     I otherwise 
# }

# requirement:
# OpenCV, Numpy

# Step:
# install : pip install numpy matplotlib opencv-python

import numpy as np
import matplotlib.pyplot as plt
import cv2, os

# bg_img = backgroudn image
# input_img = object in background 
def computeDifference(bg_img, input_img):
    # tính toán giá trị khác biệt tuyệt đối giữa 2 hình ảnh 
    diff_three_channel = np.abs(bg_img - input_img)

    # chuyển dổi sự khác biệt 3 kênh bằng cách tính trung bình 
    # Điều này giúp giảm dữ liệu khác biệt từ 3 kênh xống còn một giá trị cường độ đặc biệt đơn lẻ 
    diff_single_channel = np.sum(diff_three_channel, axis=2) / 3.0

    # Chuyển đổi trở lại giá trị sang 'uint8' 0-256 để thao tác với ảnh
    diff_single_channel = diff_single_channel.astype('uint8')

    return diff_single_channel



def computeBinaryMask(difference_single_channel, threshold=15):
    # Dùng ngưỡng để tạo mask nhị phân 
    # Các điểm ảnh có sự chênh lệch màu sắc lớn hơn hoặc bằng 15 sẽ được đặt là 255 (trắng)
    # và 0 (Đen)
    difference_single_channel = np.where(difference_single_channel >= threshold, 255, 0)

    # Ghép nối mask nhị phân thành 3 kênh để phù hợp với định dạng RGB
    binary_binary = np.stack((difference_single_channel, )*3, axis=-1)

    return binary_binary

# main func
def replaceBackGround(bg_image1, bg_image2, ob_image):
    # Tính toán sự khác biệt giữa hình ảnh và hình nền thứ nhất
    diff_single_channel = computeDifference(bg_image1, ob_image)

    # Tính toán mark dựa trên threshold 
    binary_mask = computeBinaryMask(difference_single_channel=diff_single_channel)

    # Thay thế hình nền:
    # ở vị trí mà mask là trắng (255): thay thế bằng hình nền thứ 2
    # ở vị trí mà mask là đen (0): giữ đối tượng

    output = np.where(binary_mask == 255, ob_image, bg_image2)

    return output

if __name__ == "__main__":
    if not all(os.path.exists(f) for f in ['background1.png', 'background2.png', 'object.png']):
        print("Lỗi: Thiếu file ảnh đầu vào!")
        exit()

    bg_image1 = cv2.imread('background1.png')
    bg_image2 = cv2.imread('background2.png')

    ob_image = cv2.imread('object.png')

    # resize image in the same size: 
    # convert BGR to RGB
    # cv2 load image default is BGR
    bg_image1 = cv2.resize(bg_image1, (640, 480))
    bg_image2 = cv2.resize(bg_image2, (640, 480))

    ob_image = cv2.resize(ob_image, (640, 480))

    # convert BRG TO RGB

    bg_image1 = cv2.cvtColor(bg_image1, cv2.COLOR_BGR2RGB)
    bg_image2 = cv2.cvtColor(bg_image2, cv2.COLOR_BGR2RGB)
    ob_image = cv2.cvtColor(ob_image, cv2.COLOR_BGR2RGB)

    # using replaceBackground() 

    output_image = replaceBackGround(bg_image1, bg_image2, ob_image)
    # Save
    output = replaceBackGround(bg_image1, bg_image2, ob_image)
    cv2.imwrite('output.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print("Save to: 'output.png'")
