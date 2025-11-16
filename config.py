# config.py

import os

# Base dataset folder
BASE_DATASET_PATH = r"C:\Users\ARTI\OneDrive\Documents\pytonCodes\cd\ShanghaiTech"

PART_A_TRAIN = os.path.join(BASE_DATASET_PATH, "part_A", "train_data")
PART_A_TEST  = os.path.join(BASE_DATASET_PATH, "part_A", "test_data")

PART_B_TRAIN = os.path.join(BASE_DATASET_PATH, "part_B", "train_data")
PART_B_TEST  = os.path.join(BASE_DATASET_PATH, "part_B", "test_data")

# Preprocessing settings
IMG_SIZE = (256, 256)
DENSITY_MODE = "adaptive"   # or "fixed"
FIXED_SIGMA = 15

# Training settings (for main.py test)
BATCH_SIZE = 2
