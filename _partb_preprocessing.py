import os
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# CHANGE THIS PATH ONLY
BASE_DIR = r"archive (2)/ShanghaiTech"

# ImageNet normalization (VGG backbone)
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def preprocess(part, split):
    image_dir = os.path.join(BASE_DIR, part, split, "images")
    gt_dir    = os.path.join(BASE_DIR, part, split, "ground-truth")

    # SAFE OUTPUT DIRECTORY (OUTSIDE ORIGINAL DATA)
    save_dir = os.path.join(BASE_DIR, "preprocessed", part, split)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nProcessing {part.upper()} | {split}")

    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.endswith(".jpg"):
            continue

        # READ IMAGE
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # NORMALIZE
        img = (img - MEAN) / STD

        # LOAD GT 
        gt_name = "GT_" + img_name.replace(".jpg", ".mat")
        gt_path = os.path.join(gt_dir, gt_name)
        mat = loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]

        #DENSITY MAP 
        h, w, _ = img.shape
        density = np.zeros((h, w), dtype=np.float32)

        for x, y in points:
            x = min(int(x), w - 1)
            y = min(int(y), h - 1)
            density[y, x] += 1

        density = gaussian_filter(density, sigma=1)

        # DOWNSAMPLE (CSRNet)
        h8, w8 = h // 8, w // 8
        density_8 = cv2.resize(density, (w8, h8), interpolation=cv2.INTER_CUBIC)
        density_8 *= 64  # preserve count

        # TORCH TENSORS
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        gt_tensor  = torch.from_numpy(density_8).unsqueeze(0).float()

        # SAVE
        save_path = os.path.join(save_dir, img_name.replace(".jpg", ".pt"))
        torch.save(
            {"image": img_tensor, "gt": gt_tensor},
            save_path,
            _use_new_zipfile_serialization=False  # WINDOWS FIX
        )


# RUN FOR ALL PARTS
if __name__ == "__main__":

    preprocess("part_A", "train_data")
    preprocess("part_A", "test_data")

    preprocess("part_B", "train_data")
    preprocess("part_B", "test_data")

    print("\n ALL PREPROCESSING DONE SUCCESSFULLY")
