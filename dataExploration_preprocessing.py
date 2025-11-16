import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob

# ==============================
# 1. Utility Functions
# ==============================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_gt_points(mat_path):
    mat = loadmat(mat_path)
    pts = mat["image_info"][0][0][0][0][0]
    return pts

def generate_density_map(img, points, sigma=4):
    density = np.zeros(img.shape[:2], dtype=np.float32)
    if len(points) == 0:
        return density

    h, w = density.shape
    for point in points:
        x = min(w-1, max(0, int(point[0])))
        y = min(h-1, max(0, int(point[1])))
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density

# ==============================
# 2. Visualization Helpers
# ==============================
def visualize_sample(img_path, mat_path):
    img = load_image(img_path)
    pts = load_gt_points(mat_path)
    density = generate_density_map(img, pts)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Image: {os.path.basename(img_path)}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.scatter(pts[:, 0], pts[:, 1], s=8, c="red")
    plt.title("Annotated Points")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(density, cmap="jet")
    plt.title(f"Density Map (Count={density.sum():.2f})")
    plt.axis("off")

    plt.show()

def plot_count_histogram(gt_paths, title):
    counts = []
    for mat_path in gt_paths:
        pts = load_gt_points(mat_path)
        counts.append(len(pts))

    plt.figure(figsize=(7, 4))
    plt.hist(counts, bins=30)
    plt.title(title)
    plt.xlabel("Crowd Count")
    plt.ylabel("Frequency")
    plt.show()

# ==============================
# 3. Dataset Class
# ==============================
class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), visualize=False):
        self.img_paths = sorted(glob(os.path.join(root_dir, "images", "*.jpg")))
        self.gt_paths = sorted(glob(os.path.join(root_dir, "ground-truth", "*.mat")))
        self.img_size = img_size
        self.visualize = visualize

        assert len(self.img_paths) == len(self.gt_paths), "Image/Label mismatch"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx])
        pts = load_gt_points(self.gt_paths[idx])
        density = generate_density_map(img, pts)

        # resize
        img = cv2.resize(img, self.img_size)
        density = cv2.resize(density, self.img_size)

        # normalize density map
        density = density * (density.sum() / max(density.sum(), 1e-8))

        # convert to tensor
        img = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)
        density = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        if self.visualize and idx == 0:
            plt.imshow(img.permute(1, 2, 0))
            plt.title("Sample Preprocessed Image")
            plt.show()

        return img, density

# ==============================
# 4. Dataset Exploration
# ==============================
def explore_dataset(base_path):
    print("Exploring dataset...")

    parts = ["part_A", "part_B"]
    sets = ["train_data", "test_data"]

    for part in parts:
        for split in sets:
            img_dir = os.path.join(base_path, part, split, "images")
            gt_dir = os.path.join(base_path, part, split, "ground-truth")

            img_paths = glob(os.path.join(img_dir, "*.jpg"))
            gt_paths = glob(os.path.join(gt_dir, "*.mat"))

            print(f"[{part}/{split}] Images: {len(img_paths)}, GT files: {len(gt_paths)}")

            # plot histogram only for train set
            if split == "train_data" and len(gt_paths) > 0:
                plot_count_histogram(gt_paths, 
                    title=f"Histogram for {part.upper()} - Train")

            # visualize one sample
            if len(img_paths) > 0:
                print(f"Visualizing sample for {part}/{split} ...")
                visualize_sample(img_paths[0], gt_paths[0])

# ==============================
# 5. Example Usage
# ==============================
if __name__ == "__main__":
    DATASET_ROOT = "ShanghaiTech"

    # Step 1: Explore dataset
    explore_dataset(DATASET_ROOT)

    # Step 2: Load dataset
    train_A = ShanghaiTechDataset(
        root_dir="ShanghaiTech/part_A/train_data",
        img_size=(256, 256),
        visualize=True
    )

    print("Length of Part A Train Dataset:", len(train_A))

    # Step 3: Use DataLoader
    loader = DataLoader(train_A, batch_size=2, shuffle=True)

    for imgs, dens in loader:
        print("Batch Image Shape:", imgs.shape)
        print("Batch Density Shape:", dens.shape)
        break
