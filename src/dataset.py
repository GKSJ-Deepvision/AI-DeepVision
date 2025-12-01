import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import os
import glob
import torchvision.transforms as T
import cv2  # This is OpenCV
import numpy as np

class CrowdDataset(Dataset):
    def __init__(self, img_dir, dmap_dir):
        self.img_dir = img_dir
        self.dmap_dir = dmap_dir
        
        # Find all .jpg files
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        
        # --- NEW ---
        # Define a target size for all images.
        # Must be divisible by 8 for the CSRNet model.
        self.target_width = 1024
        self.target_height = 768 # (768 is divisible by 8)
        
        # Define the transformations for the images
        self.img_transform = T.Compose([
            T.Resize((self.target_height, self.target_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])
        # -----------

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # --- Load Image ---
        img_path = self.img_files[index]
        img = Image.open(img_path).convert('RGB')
        
        # --- Load Density Map ---
        # Get the corresponding .mat file name
        mat_name = "GT_" + os.path.basename(img_path).replace('.jpg', '.mat')
        mat_path = os.path.join(self.dmap_dir, mat_name)
        
        mat = loadmat(mat_path)
        
        # --- FIX from last error ---
        # This is the correct way to access the ShanghaiTech .mat data
        dmap = mat['image_info'][0,0][0,0][1].astype(np.float32)
        
        # --- !! NEW RESIZING CODE !! ---
        
        # 1. Resize the image using the transform
        img_tensor = self.img_transform(img)

        # 2. Resize the density map
        # We must resize the density map to match the *output* of the model,
        # which is 1/8th the size of the input image.
        output_width = self.target_width // 8
        output_height = self.target_height // 8
        
        # Resize using cv2 (INTER_LINEAR is a good choice)
        dmap_resized = cv2.resize(dmap, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

        # 3. Correct the density sum
        # Resizing changes the total count. We must scale it back.
        orig_sum = np.sum(dmap)
        new_sum = np.sum(dmap_resized)
        
        if new_sum > 0: # Avoid division by zero
            dmap_resized = dmap_resized * (orig_sum / new_sum)
        
        # 4. Convert dmap to a tensor and add a channel dimension
        # [H, W] -> [1, H, W]
        dmap_tensor = torch.tensor(dmap_resized, dtype=torch.float32).unsqueeze(0)
        
        return img_tensor, dmap_tensor