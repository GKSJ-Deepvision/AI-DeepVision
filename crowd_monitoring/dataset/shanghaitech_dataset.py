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
                 fixed_sigma=15, transform=None):

        self.img_paths = sorted(glob(os.path.join(root_dir, "images", "*.jpg")))
        self.gt_paths  = sorted(glob(os.path.join(root_dir, "ground-truth", "*.mat")))

        assert len(self.img_paths) == len(self.gt_paths), "Images & GT Count Mismatch"

        self.img_size = img_size
        self.density_mode = density_mode
        self.fixed_sigma = fixed_sigma
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx])
        pts = load_gt_points(self.gt_paths[idx])

        density = generate_density_map(img, pts, mode=self.density_mode, fixed_sigma=self.fixed_sigma)

        # Resize image & density
        img_r = cv2.resize(img, self.img_size)
        den_r = cv2.resize(density, self.img_size)

        # Density normalization
        total = density.sum()
        if den_r.sum() > 0:
            den_r *= total / den_r.sum()

        img_t = torch.tensor(img_r / 255.0).permute(2, 0, 1).float()
        den_t = torch.tensor(den_r).unsqueeze(0).float()

        return img_t, den_t
