#worked
#clean_and_normalize.py
#cleans and normalizes the risized raw images folder according to the path specified
import os
import cv2
import numpy as np

# ---------- SETTINGS ----------
RESIZED_IMAGES_FOLDER = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_images_trainA"
CLEANED_OUTPUT_FOLDER = r"C:\Users\mahal\OneDrive\Desktop\DL\cleaned_resized_images_trainA"
NORMALIZED_OUTPUT_FOLDER = r"C:\Users\mahal\OneDrive\Desktop\DL\normalized_resized_images_trainA"

CLEAN_METHOD = "denoise+clahe"   # options: "none", "denoise", "clahe", "denoise+clahe"
NORMALIZE_METHOD = "minmax"      # options: "minmax", "imagenet"
# ------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)]


# ---------- CLEANING ----------
def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def clahe(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = c.apply(y)
    merged = cv2.merge((y2, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def clean_image(img, method):
    if method == "none":
        return img
    if method == "denoise":
        return denoise(img)
    if method == "clahe":
        return clahe(img)
    if method == "denoise+clahe":
        return clahe(denoise(img))
    return img


# ---------- NORMALIZATION ----------
def normalize(img_rgb, method):
    img_f = img_rgb.astype(np.float32) / 255.0
    if method == "minmax":
        return img_f
    elif method == "imagenet":
        return (img_f - IMAGENET_MEAN) / IMAGENET_STD
    return img_f


# ---------- MAIN PROCESS ----------
def clean_and_normalize():
    ensure_dir(CLEANED_OUTPUT_FOLDER)
    ensure_dir(NORMALIZED_OUTPUT_FOLDER)

    images = list_images(RESIZED_IMAGES_FOLDER)

    print(f"Found {len(images)} images.")

    for fname in images:
        path = os.path.join(RESIZED_IMAGES_FOLDER, fname)

        img = cv2.imread(path)
        if img is None:
            print("Failed to read:", fname)
            continue

        # 1. Clean image
        cleaned = clean_image(img, CLEAN_METHOD)

        # Save cleaned image
        cv2.imwrite(os.path.join(CLEANED_OUTPUT_FOLDER, fname), cleaned)

        # 2. Normalize image (convert to RGB first)
        img_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        norm = normalize(img_rgb, NORMALIZE_METHOD)

        # Save normalized .npy tensor
        base = os.path.splitext(fname)[0]
        np.save(os.path.join(NORMALIZED_OUTPUT_FOLDER, base + ".npy"), norm.astype(np.float32))

    print("\n✔ Cleaning Completed")
    print("✔ Normalization Completed")
    print("Cleaned images saved to:", CLEANED_OUTPUT_FOLDER)
    print("Normalized tensors saved to:", NORMALIZED_OUTPUT_FOLDER)


if __name__ == "__main__":
    clean_and_normalize()
