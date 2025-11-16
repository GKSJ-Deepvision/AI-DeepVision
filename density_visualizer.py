import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG (CHANGE PATHS HERE)
# -----------------------------
IMAGES_FOLDER  = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_images_trainA"
DENSITY_FOLDER = r"C:\Users\mahal\OneDrive\Desktop\DL\resized_density_trainA"

# -----------------------------
# VISUALIZER FUNCTION
# -----------------------------
def visualize(image_path, density_path):
    # check files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(density_path):
        raise FileNotFoundError(f"Density file not found: {density_path}")

    # load image (BGR → RGB)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load density map
    dens = np.load(density_path)

    # compute predicted count
    count = dens.sum()

    # create figure
    plt.figure(figsize=(10, 5))

    # original
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # density map as colored heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(dens, cmap='jet')
    plt.title(f"Density Map\nCount = {count:.1f}")
    plt.axis("off")

    # overlay heatmap on image
    heatmap_bgr = cv2.applyColorMap(
        cv2.normalize(dens, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    # convert BGR -> RGB to match img_rgb
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    # blend (img_rgb and heatmap_rgb are uint8)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay (Image + Density)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# -----------------------------
# RUN EXAMPLE
# -----------------------------
if __name__ == "__main__":
    # change to any existing file name in your folder
    fname = "IMG_1.jpg"

    image_path  = os.path.join(IMAGES_FOLDER, fname)
    density_path = os.path.join(DENSITY_FOLDER, fname.replace(".jpg", ".npy"))

    visualize(image_path, density_path)
