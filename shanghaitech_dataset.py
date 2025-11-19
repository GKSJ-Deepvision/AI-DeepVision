# dataset/shanghaitech_dataset.py

import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from .utils import load_image, load_gt_points, generate_density_map


class ShanghaiTechDataset(Dataset):

    def __init__(self, root_dir, img_size=(256, 256), density_mode="adaptive",
             fixed_sigma=15):

        self.img_paths = sorted(glob(os.path.join(root_dir, "images", "*.jpg")))
        self.gt_paths  = sorted(glob(os.path.join(root_dir, "ground-truth", "*.mat")))

        assert len(self.img_paths) == len(self.gt_paths), "Images & GT Count Mismatch"

        self.img_size = img_size
        self.density_mode = density_mode
        self.fixed_sigma = fixed_sigma

        # -------------------------
        # ImageNet Mean / Std for VGG
        # -------------------------
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load image and points
        img = load_image(self.img_paths[idx])
        pts = load_gt_points(self.gt_paths[idx])

        # 2. Generate full-resolution density map
        density = generate_density_map(img, pts, mode=self.density_mode, fixed_sigma=self.fixed_sigma)

        # 3. Resize image to 256×256
        img_resized = cv2.resize(img, self.img_size)

        # 4. Resize density map to match image size (256×256)
        density_resized = cv2.resize(density, self.img_size)

        # 5. Downsample by 8× → 32×32
        H, W = self.img_size
        down_H, down_W = H // 8, W // 8  # 32×32

        density_down = cv2.resize(density_resized, (down_W, down_H))

        # 6. Multiply by 64 to preserve total count
        density_down *= (8 * 8)

            # Convert to tensors
        # -------------------------------
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()

        # ImageNet normalization
        img_tensor = (img_tensor - self.mean) / self.std

        density_tensor = torch.tensor(density_down).unsqueeze(0).float()

        return img_tensor, density_tensor
