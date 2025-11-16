# dataset/visualization.py
import matplotlib.pyplot as plt
from .utils import load_image, load_gt_points, generate_density_map


def visualize_sample(img_path, gt_path):
    img = load_image(img_path)
    points = load_gt_points(gt_path)
    density = generate_density_map(img, points)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], s=5, c="red")
    plt.title("Annotations")

    plt.subplot(1, 3, 3)
    plt.imshow(density, cmap="jet")
    plt.title("Density Map")

    plt.show()


def plot_count_histogram(gt_paths, title="Crowd Count Histogram"):
    counts = []
    for gt in gt_paths:
        pts = load_gt_points(gt)
        counts.append(len(pts))

    plt.hist(counts, bins=30)
    plt.title(title)
    plt.xlabel("Crowd Count")
    plt.ylabel("Frequency")
    plt.show()
