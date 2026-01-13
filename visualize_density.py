import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# ---- PATHS ----
img_path = r"C:/Users/Lenovo/Downloads/deep dataset/part_A_final/test_data/images/IMG_1.jpg"
gt_path  = r"C:/Users/Lenovo/Downloads/deep dataset/part_A_final/test_data/ground_truth/GT_IMG_1.mat"

# ---- LOAD GT FILE ----
data = scipy.io.loadmat(gt_path)
points = data['image_info'][0][0][0][0][0]
print("Points loaded:", len(points))

def create_density_map(img, points, sigma=15):
    density = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for p in points:
        y = min(int(p[1]), img.shape[0] - 1)
        x = min(int(p[0]), img.shape[1] - 1)
        density[y, x] = 1

    density = gaussian_filter(density, sigma=sigma)
    return density

   # ---- LOAD IMAGE ----
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- CREATE DENSITY MAP ----
density = create_density_map(img, points)

# ---- VISUALIZE ----
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(density, cmap="jet")
plt.title("Density Map")
plt.axis("off")

plt.show()