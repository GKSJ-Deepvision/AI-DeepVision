import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# --------------------------------------------------
# 1. Load Image & Ground Truth
# --------------------------------------------------
img_path = r"archive (2)\ShanghaiTech\part_B\train_data\images\IMG_5.jpg"
gt_path = r"archive (2)\ShanghaiTech\part_B\train_data\ground-truth\GT_IMG_5.mat"

# Load image in RGB
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load .mat ground truth
gt = loadmat(gt_path)
points = gt["image_info"][0][0][0][0][0]   # (N, 2) array

print("Original no. of points:", len(points))

# --------------------------------------------------
# 2. Resize image and scale GT points
# --------------------------------------------------
resize_shape = (256, 256)   # (width, height)

orig_h, orig_w, _ = img.shape
img_resized = cv2.resize(img, resize_shape)

w_ratio = resize_shape[0] / orig_w
h_ratio = resize_shape[1] / orig_h

# scale GT point coordinates
points_rescaled = points * [w_ratio, h_ratio]

# --------------------------------------------------
# 3. Normalize image
# --------------------------------------------------
img_normalized = img_resized / 255.0

# Convert to tensor (C, H, W)
img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)
print("Image tensor shape:", img_tensor.shape)

# --------------------------------------------------
# 4. Density-map generation function
# --------------------------------------------------
def generate_density_map(image, points, sigma=4):
    """
    Generates density map where each head annotation gets a Gaussian.
    Lower sigma helps highlight individual heads more clearly.
    """
    h, w, _ = image.shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    # place a "1" at each GT head location
    for x, y in points:
        x = int(min(w - 1, max(0, x)))
        y = int(min(h - 1, max(0, y)))
        density[y, x] = 1

    # blur the 1-pixel dots into smooth Gaussians
    density = gaussian_filter(density, sigma=sigma)

    return density

# --------------------------------------------------
# 5. Generate density map
# --------------------------------------------------
density_map = generate_density_map(img_resized, points_rescaled, sigma=4)
density_tensor = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

print("Density map shape:", density_tensor.shape)
print("Estimated Count (sum of density map):", float(np.sum(density_map)))

# --------------------------------------------------
# 6. Plot everything
# --------------------------------------------------
plt.figure(figsize=(6,6))
plt.imshow(img_resized)
plt.title("Resized Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(density_map, cmap='jet')
plt.colorbar(label="Density")
plt.title("Density Map (sigma=4)")
plt.axis("off")
plt.show()