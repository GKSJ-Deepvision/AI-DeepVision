# dataset/utils.py
import os
import glob
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt_points(gt_path):
    mat = loadmat(gt_path)

    # Correct structure for ShanghaiTech
    try:
        pts = mat["image_info"][0][0][0][0][0]
    except:
        raise ValueError(f"Invalid .mat structure: {gt_path}")

    return np.array(pts, dtype=np.float32)


def list_image_gt_pairs(root_dir):
    img_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpg")))
    gt_paths  = sorted(glob.glob(os.path.join(root_dir, "ground-truth", "*.mat")))
    return img_paths, gt_paths


def save_density_map(path, density):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, density.astype(np.float32))


def generate_density_map(img, points, mode="adaptive", fixed_sigma=15):
    h, w = img.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    if mode == "fixed":
        for x, y in points:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                density[yi, xi] += 1
        density = gaussian_filter(density, fixed_sigma)
        return density

    # Adaptive mode (simple version)
    from scipy.spatial import KDTree
    tree = KDTree(points)

    for point in points:
        dist, idx = tree.query(point, k=4)
        sigma = np.mean(dist[1:]) * 0.3

        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1

    density = gaussian_filter(density, sigma)
    return density
