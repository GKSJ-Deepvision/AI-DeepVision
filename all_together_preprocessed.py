import os
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------- PATHS ----------
image_dir = r"archive (2)/ShanghaiTech/part_B/train_data/images"
gt_dir    = r"archive (2)/ShanghaiTech/part_B/train_data/ground-truth"

# ---------- NEW OUTPUT FOLDERS (SAFE) ----------
save_img_dir = os.path.join(image_dir, "images_prepro")
os.makedirs(save_img_dir, exist_ok=True)

save_gt_dir = os.path.join(gt_dir, "ground_truth_prepro")
os.makedirs(save_gt_dir, exist_ok=True)

# ImageNet normalization (for VGG)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

# ---------- LOOP OVER ALL IMAGES ----------
for img_name in tqdm(os.listdir(image_dir)):

    if not img_name.endswith(".jpg"):
        continue

    # ---- 1. Read Image ----
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    # ---- 2. Normalize ----
    img_norm = (img - mean) / std

    # ---- 3. Load GT points ----
    gt_name = "GT_" + img_name.replace(".jpg", ".mat")
    gt_path = os.path.join(gt_dir, gt_name)

    mat = loadmat(gt_path)
    points = mat["image_info"][0][0][0][0][0]

    # ---- 4. Create density ----
    density = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for x, y in points:
        x = min(int(x), img.shape[1] - 1)
        y = min(int(y), img.shape[0] - 1)
        density[y, x] = 1

    density = gaussian_filter(density, sigma=1)

    # ---- 5. Downsample by 8x ----
    h8, w8 = img.shape[0] // 8, img.shape[1] // 8
    density_8 = cv2.resize(density, (w8, h8), interpolation=cv2.INTER_CUBIC)
    density_8 *= 64

    # ---- 6. Convert to tensors ----
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)
    gt_tensor  = torch.from_numpy(density_8).unsqueeze(0)

    # ---- 7. SAVE IN NEW FOLDER (SAFE) ----
    save_name = img_name.replace(".jpg", "_prepro.pt")

    torch.save(
        {"image": img_tensor, "gt": gt_tensor},
        os.path.join(save_img_dir, save_name)
    )
