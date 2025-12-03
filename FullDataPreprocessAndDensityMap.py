import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# Load Image
img_path = r"archive (2)\\ShanghaiTech\\part_A\\train_data\\images\\IMG_5.jpg"
gt_path  = r"archive (2)\\ShanghaiTech\\part_A\\train_data\\ground-truth\\GT_IMG_5.mat"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img.astype(np.float32) / 255.0   # scale 0–1

# Normalize ImageNet (VGG)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
img_norm = (img_float - mean) / std

# Load GT Points
mat = loadmat(gt_path)
points = mat["image_info"][0][0][0][0][0]   # (N,2) points

#Create Simple Density (Optional Gaussian)
density = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
for x, y in points:
    x = min(int(x), img.shape[1] - 1)
    y = min(int(y), img.shape[0] - 1)
    density[y, x] = 1
density = gaussian_filter(density, sigma=7)

#Downsample by 8x + Multiply by 64
h, w = density.shape
density_8 = cv2.resize(density, (w // 8, h // 8))
density_8 = density_8 * 64   # preserve count

#Convert to Tensors
img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)   # CHW
gt_tensor  = torch.from_numpy(density_8).unsqueeze(0)       # 1×H/8×W/8

print("Image tensor shape:", img_tensor.shape)
print("GT tensor shape:", gt_tensor.shape)

# Show first 5 rows of tensor
print("\nFirst 5 rows of image tensor (channel 0):")
print(img_tensor[0, :5, :5])   # first 5×5 region

#Visualization
# 1. Image with GT points overlaid
img_with_points = img.copy()
for x, y in points:
    cv2.circle(img_with_points, (int(x), int(y)), 3, (255, 0, 0), -1)
# 2. Show all results
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Image + GT Points")
plt.imshow(img_with_points)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Downsampled GT Heatmap (8x)")
plt.imshow(density_8, cmap="jet")
plt.axis("off")

plt.show()
