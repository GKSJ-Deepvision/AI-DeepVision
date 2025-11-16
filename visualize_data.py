import cv2
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

IMAGE_PATH = "data/ShanghaiTech/part_A/train_data/images/IMG_1.jpg"
MAT_PATH   = "data/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_1.mat"
OUTPUT_DIR = "visualization_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_points(mat_path):
    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0][0][0][0][0]   
    return points


def create_impulse_map(points, h, w):
    impulse = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        if 0 <= int(y) < h and 0 <= int(x) < w:
            impulse[int(y), int(x)] = 1
    return impulse


def gaussian_density_map(impulse, sigma=4):
    density = gaussian_filter(impulse, sigma=sigma)
    return density


def show_image_with_points(image, points):
    img_copy = image.copy()
    for x, y in points:
        cv2.circle(img_copy, (int(x), int(y)), 3, (255, 0, 0), -1)
    return img_copy


def overlay_heatmap(image, density):
    heatmap = cv2.applyColorMap((density / density.max() * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


if __name__ == "__main__":

    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    points = load_points(MAT_PATH)

    # Image with head points
    img_with_points = show_image_with_points(img_rgb, points)
    plt.imshow(img_with_points)
    plt.title("Original Image with Head Point Annotations")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_img_with_points.png"))
    plt.close()

    # Impulse Map
    impulse_map = create_impulse_map(points, h, w)
    plt.imshow(impulse_map, cmap="gray")
    plt.title("Impulse Map")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "2_impulse_map.png"))
    plt.close()

    # Density Map
    density_map = gaussian_density_map(impulse_map)
    plt.imshow(density_map, cmap="jet")
    plt.title(f"Density Map (Count={density_map.sum():.2f})")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "3_density_map.png"))
    plt.close()

    # Heat map overlay 
    overlay_img = overlay_heatmap(img_rgb, density_map)
    plt.imshow(overlay_img)
    plt.title("Heatmap Overlay")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "4_heatmap_overlay.png"))
    plt.close()

    print("\nVisualization completed! Check folder: visualization_output\n")
