#worked 
#generates density maps according to the folders mentioned
import os
import numpy as np
import scipy.io as sio
import cv2
from scipy.ndimage import gaussian_filter

# -----------------------------
# CONFIGURATION
# -----------------------------

RESIZED_IMAGES = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_images_trainA"
RESIZED_GTS    = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_ground_truth_trainA"
OUTPUT_FOLDER  = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_density_trainA"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def load_coordinates(mat_path):
    """Load dot coordinates from ShanghaiTech-style .mat file"""
    mat = sio.loadmat(mat_path)
    coords = mat["image_info"][0][0][0][0]   # shape (N, 2)
    return coords


def generate_density_map(img_shape, points, sigma=4):
    """
    Create a density map of same height×width as image.
    points: (N, 2) array → x,y coordinates
    """
    H, W = img_shape[:2]
    density = np.zeros((H, W), dtype=np.float32)

    for p in points:
        x = int(min(W - 1, max(0, p[0])))
        y = int(min(H - 1, max(0, p[1])))
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density


# -----------------------------
# MAIN PROCESS
# -----------------------------

img_files = sorted(os.listdir(RESIZED_IMAGES))

for fname in img_files:

    if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        continue

    img_path = os.path.join(RESIZED_IMAGES, fname)
    gt_path = os.path.join(RESIZED_GTS, f"GT_{fname.split('.')[0]}.mat")

    if not os.path.exists(gt_path):
        print("Missing GT for:", fname)
        continue

    # load resized image
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # load coords from resized .mat file
    points = load_coordinates(gt_path)

    # generate density
    density = generate_density_map((H, W, 3), points, sigma=4)

    # save density as .npy
    out_name = os.path.splitext(fname)[0] + ".npy"
    np.save(os.path.join(OUTPUT_FOLDER, out_name), density.astype(np.float32))

    print("Saved density for:", fname)

print("\n✔ All density maps created successfully!")
print("Saved to:", OUTPUT_FOLDER)
