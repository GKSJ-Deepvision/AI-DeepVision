#worked
#resize_images_and_annotations.py
#risizing the images and grothtruth files using folder paths
import os
import cv2
import numpy as np
import scipy.io as sio

# ------------------ CONFIG ------------------
TARGET_W = 512   # change if you want a different size
TARGET_H = 512
# --------------------------------------------

# ---------- YOUR PROVIDED PATHS (TEST DATA) ----------
IMG_DIR = r"C:\Users\mahal\OneDrive\Desktop\DL\archive\ShanghaiTech\part_A\train_data\images"
GT_DIR  = r"C:\Users\mahal\OneDrive\Desktop\DL\archive\ShanghaiTech\part_A\train_data\ground-truth"
OUT_IMG_DIR = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_images_trainA"
OUT_GT_DIR  = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_ground_truth_trainA"
# ----------------------------------------------------

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def list_images(folder):
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_points_from_mat(mat_path):
    # ShanghaiTech typical format: mat["image_info"][0][0][0][0][0] -> Nx2 (x,y)
    mat = sio.loadmat(mat_path)
    try:
        pts = mat["image_info"][0][0][0][0][0]
        return pts.astype(np.float32)
    except Exception:
        # fallback: try direct 'gt' or other common keys, otherwise return empty
        for key in mat:
            if isinstance(mat[key], np.ndarray) and mat[key].ndim == 2 and mat[key].shape[1] == 2:
                return mat[key].astype(np.float32)
        return np.zeros((0, 2), dtype=np.float32)

def save_points_to_mat(points, out_path):
    data = {"image_info": [[[[points]]]]}
    sio.savemat(out_path, data)

def resize_points(pts, sx, sy):
    if pts.size == 0:
        return pts
    out = np.zeros_like(pts, dtype=np.float32)
    out[:, 0] = pts[:, 0] * sx
    out[:, 1] = pts[:, 1] * sy
    return out

def resize_dataset(img_dir, gt_dir, out_img_dir, out_gt_dir, target_w, target_h):
    ensure_dir(out_img_dir)
    ensure_dir(out_gt_dir)

    img_files = list_images(img_dir)
    if not img_files:
        print("No images found in", img_dir)
        return

    for i, fname in enumerate(img_files):
        base = os.path.splitext(fname)[0]

        img_path = os.path.join(img_dir, fname)
        # ShanghaiTech ground-truth name is usually "GT_IMG_XXXX.mat" matching image "IMG_XXXX.jpg"
        gt_name = f"GT_{base}.mat"
        gt_path = os.path.join(gt_dir, gt_name)

        # fallback to basename.mat if GT file differs
        if not os.path.exists(gt_path):
            alt = os.path.join(gt_dir, base + ".mat")
            if os.path.exists(alt):
                gt_path = alt
            else:
                print(f"[{i+1}/{len(img_files)}] Missing GT for: {fname}  -> skipping")
                continue

        # load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{i+1}/{len(img_files)}] Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        sx = target_w / float(w)
        sy = target_h / float(h)

        # resize image
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # load and resize GT points
        pts = read_points_from_mat(gt_path)
        resized_pts = resize_points(pts, sx, sy)

        # save resized image
        out_img_path = os.path.join(out_img_dir, fname)
        cv2.imwrite(out_img_path, resized_img)

        # save resized .mat using same GT naming convention
        out_mat_name = f"GT_{base}.mat"
        out_mat_path = os.path.join(out_gt_dir, out_mat_name)
        save_points_to_mat(resized_pts, out_mat_path)

        print(f"[{i+1}/{len(img_files)}] Resized: {fname}  -> points: {resized_pts.shape[0]}")

    print("\n✔ All done. Resized images saved to:", out_img_dir)
    print("✔ Resized ground-truth saved to:", out_gt_dir)

if __name__ == "__main__":
    resize_dataset(IMG_DIR, GT_DIR, OUT_IMG_DIR, OUT_GT_DIR, TARGET_W, TARGET_H)
