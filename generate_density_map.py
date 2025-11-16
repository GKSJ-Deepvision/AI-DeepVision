import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

# --- STEP 1: Load one sample image ---
image_path = r"C:\Users\nshre\OneDrive\Desktop\NITHYA-SHREE\Crowd\sample.jpg"  # Change filename to one that exists
img = np.array(Image.open(image_path).convert('RGB'))

# --- STEP 2: Example Ground Truth Coordinates (x, y) ---
annotations = np.array([
    [150, 200],
    [220, 250],
    [300, 320],
    [350, 280],
    [400, 350],
])  # Example: 5 people

# --- STEP 3: Initialize an empty density map ---
density_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

# --- STEP 4: Mark each annotation point on the map ---
for point in annotations:
    x, y = min(int(point[0]), img.shape[1]-1), min(int(point[1]), img.shape[0]-1)
    density_map[y, x] = 1

# --- STEP 5: Apply Gaussian filter to create smooth density ---
density_map = gaussian_filter(density_map, sigma=15)

# --- STEP 6: Visualize image and density map ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(density_map, cmap='jet')
plt.title("Ground Truth Density Map")
plt.axis("off")

plt.show()

# --- STEP 7: Save density map ---
save_path = r"C:\Users\nshre\OneDrive\Desktop\NITHYA-SHREE\density_map.npy"
np.save(save_path, density_map)
print(f"✅ Density map saved to: {save_path}")
