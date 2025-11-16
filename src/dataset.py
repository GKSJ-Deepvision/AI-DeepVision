import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from .preprocess import preprocess_pipeline

class CrowdDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img_tensor = preprocess_pipeline(img_path)

        mat_path = img_path.replace(".jpg", ".mat")
        mat = loadmat(mat_path)

        if "density" in mat:
            dmap = mat["density"]
        else:
            dmap = mat[list(mat.keys())[-1]]

        dmap = dmap.astype(np.float32)
        dmap_tensor = torch.from_numpy(dmap).unsqueeze(0)

        return img_tensor, dmap_tensor
