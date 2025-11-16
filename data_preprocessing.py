import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

IMAGE_DIR = r"data/ShanghaiTech/part_A/train_data/images"
GT_DIR    = r"data//ShanghaiTech/part_A/train_data/ground-truth"

OUT_IMAGE_DIR   = r"data//ShanghaiTech/part_A/train_data/processed_images"
OUT_DENSITY_DIR = r"data//ShanghaiTech/part_A/train_data/density_maps"

# image resize dimensions
TARGET_H = 512
TARGET_W = 512

GAUSSIAN_SIGMA = 4.0    # Gaussian sigma value for density map

def load_points_from_mat(mat_path):
    """
    Loads head point coordinates (x, y) from a GT_*.mat file.
    For ShanghaiTech format: mat['image_info'][0][0][0][0][0]
    returns an (N, 2) numpy array.
    """
    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]  # points shape (N, 2)
    return points


def generate_density_map(points, out_h, out_w):
    """
    Creates a density map of shape (out_h, out_w) from head points.
    `points` are assumed to already be scaled to this resolution.
    """
    density = np.zeros((out_h, out_w), dtype=np.float32)

    if len(points) == 0:
        return density

    # Put an impulse (1) at each head location
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < out_w and 0 <= y < out_h:
            density[y, x] += 1.0

    # Apply Gaussian filter to spread each point
    density = gaussian_filter(density, sigma=GAUSSIAN_SIGMA)

    return density


def preprocess_single_image(img_name):
    """
    Preprocess one image + its ground-truth file:
    - load image
    - resize
    - convert BGR->RGB
    - load & scale points
    - generate density map
    - save processed image and density map
    """

    img_path = os.path.join(IMAGE_DIR, img_name)
    base_name = os.path.splitext(img_name)[0] 
    mat_name  = f"GT_{base_name}.mat"
    gt_path   = os.path.join(GT_DIR, mat_name)

    if not os.path.exists(gt_path):
        print(f"[WARNING] Ground-truth not found for {img_name}: {gt_path}")
        return

    # load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARNING] Failed to read image: {img_path}")
        return

    orig_h, orig_w = img_bgr.shape[:2]

    # Resize image to fixed size 512x512
    img_bgr_resized = cv2.resize(img_bgr, (TARGET_W, TARGET_H))

    # Convert BGR -> RGB 
    img_rgb_resized = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)

    # Normalize (0–1) 
    img_normalized = img_rgb_resized.astype(np.float32) / 255.0

    # Load ground truth points 
    points = load_points_from_mat(gt_path)  # (N, 2)

    # Scale points according to new image size
    # Scale factors from original -> resized
    scale_x = TARGET_W / orig_w
    scale_y = TARGET_H / orig_h

    if len(points) > 0:
        points_rescaled = np.zeros_like(points, dtype=np.float32)
        points_rescaled[:, 0] = points[:, 0] * scale_x  # x
        points_rescaled[:, 1] = points[:, 1] * scale_y  # y
    else:
        points_rescaled = points

    # Generate density map on resized grid-
    density = generate_density_map(points_rescaled, TARGET_H, TARGET_W)

    # Save processed image and density 
    # Ensure output dirs exist
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_DENSITY_DIR, exist_ok=True)

    # Save RGB image
    img_bgr_save = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)
    out_img_path = os.path.join(OUT_IMAGE_DIR, img_name)
    cv2.imwrite(out_img_path, img_bgr_save)

    # Save density map as .npy
    out_density_name = base_name + ".npy"
    out_density_path = os.path.join(OUT_DENSITY_DIR, out_density_name)
    np.save(out_density_path, density)

    print(f"[OK] Processed {img_name} -> {out_img_path}, {out_density_path} "
          f"(count ≈ {density.sum():.2f})")


def preprocess_all():
    """
    Loops over all images in IMAGE_DIR and preprocesses each one.
    """
    img_files = [f for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(".jpg") or f.lower().endswith(".png")]

    img_files.sort()

    print(f"Found {len(img_files)} images in {IMAGE_DIR}")

    for img_name in img_files:
        preprocess_single_image(img_name)

    print("Preprocessing completed for all images.")


if __name__ == "__main__":
    preprocess_all()
