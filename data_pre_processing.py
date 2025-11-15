import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Path to image and its corresponding .mat ground-truth
img_path = r"archive (2)\ShanghaiTech\part_B\train_data\images\IMG_6.jpg"
gt_path = r"archive (2)\ShanghaiTech\part_B\train_data\ground-truth\GT_IMG_6.mat"

# Load the image using OpenCV and convert to RGB (OpenCV loads as BGR by default)
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

resize_shape = (256, 256)
img_resized = cv2.resize(img, resize_shape)
img_normalized = img_resized / 255.0   # normalization

plt.imshow(img_normalized)
plt.title("Resized & Normalized Image")
plt.show()

print("New shape:", img_normalized.shape)
print("Pixel range:", img_normalized.min(), "to", img_normalized.max())