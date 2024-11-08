import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Đọc ảnh từ cùng thư mục chứa file code
image_path = os.path.join(os.getcwd(), 'anhvetinh.jpeg')
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")

# 1. Bộ lọc Sobel
sobel_x = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# 2. Bộ lọc Prewitt
prewitt_x = cv2.filter2D(image, ddepth=-1, kernel=np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitt_y = cv2.filter2D(image, ddepth=-1, kernel=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
prewitt = np.sqrt(prewitt_x.astype(np.float64)**2 + prewitt_y.astype(np.float64)**2)

# 3. Bộ lọc Roberts
roberts_x = cv2.filter2D(image, ddepth=-1, kernel=np.array([[1, 0], [0, -1]]))
roberts_y = cv2.filter2D(image, ddepth=-1, kernel=np.array([[0, 1], [-1, 0]]))
roberts = np.sqrt(roberts_x.astype(np.float64)**2 + roberts_y.astype(np.float64)**2)

# 4. Gaussian Blur
gaussian = cv2.GaussianBlur(image, (5, 5), 0)

# 5. Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Hiển thị kết quả
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'Gaussian', 'Canny']
images = [image, sobel, prewitt, roberts, gaussian, canny]

for i, ax in enumerate(axs.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(titles[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
