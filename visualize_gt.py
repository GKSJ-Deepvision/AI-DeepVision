import os
import cv2
import scipy.io
import matplotlib.pyplot as plt

# Correct dataset paths
img_path = r"C:\Users\bolak\OneDrive\Desktop\deepvision\deepvision\ShanghaiTech\part_A\train_data\images"
gt_path = r"C:\Users\bolak\OneDrive\Desktop\deepvision\deepvision\ShanghaiTech\part_A\train_data\ground-truth"

# Pick first image
first_img = os.listdir(img_path)[0]

# Image path
img_full = os.path.join(img_path, first_img)

# Corresponding ground-truth file: ADD "GT_" prefix
mat_name = "GT_" + first_img.replace(".jpg", ".mat")
mat_full = os.path.join(gt_path, mat_name)

print("Image file:", img_full)
print("GT mat file:", mat_full)

# Load image
img = cv2.imread(img_full)
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    raise ValueError("Image not found or unable to load.")

# Load mat
mat = scipy.io.loadmat(mat_full)

# Extract head points
try:
    points = mat["image_info"][0][0][0][0][0]
except Exception:
    points = mat["image_info"][0][0][0][0][1]

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1], s=12, color='red')
plt.title("Ground Truth Points (Crowd Heads)")
plt.axis("off")
plt.show()
