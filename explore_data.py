import os

# Define dataset paths
root = r"archive (2)\\ShanghaiTech"
partA_train = os.path.join(root, "part_A", "train_data")
partA_test  = os.path.join(root, "part_A", "test_data")
partB_train = os.path.join(root, "part_B", "train_data")
partB_test  = os.path.join(root, "part_B", "test_data")

# Count images and ground truths in each part
def count_files(folder):
    img_dir = os.path.join(folder, "images")
    gt_dir  = os.path.join(folder, "ground-truth")
    return len(os.listdir(img_dir)), len(os.listdir(gt_dir))

print("Part A - Train:", count_files(partA_train))
print("Part A - Test :", count_files(partA_test))
print("Part B - Train:", count_files(partB_train))
print("Part B - Test :", count_files(partB_test))


import cv2
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

def show_random_images(dataset_path, n=4):
    img_dir = os.path.join(dataset_path, "images")
    gt_dir = os.path.join(dataset_path, "ground-truth")
    files = os.listdir(img_dir)
    samples = random.sample(files, n)

    plt.figure(figsize=(12, 8))
    for i, img_name in enumerate(samples):
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, "GT_" + img_name.split(".jpg")[0] + ".mat")

        # Load image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt = loadmat(gt_path)
        points = gt["image_info"][0][0][0][0][0]
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(f"{img_name}\nPeople Count: {len(points)}")
        plt.axis("off")
    plt.show()

show_random_images(partB_train, 4)


import numpy as np
import tqdm

def get_counts(dataset_path):
    gt_dir = os.path.join(dataset_path, "ground-truth")
    gt_files = os.listdir(gt_dir)
    counts = []
    for file in tqdm.tqdm(gt_files):
        data = loadmat(os.path.join(gt_dir, file))
        points = data["image_info"][0][0][0][0][0]
        counts.append(len(points))
    return counts
counts_partB_train = get_counts(partB_train)

print(f"Total images: {len(counts_partB_train)}")
print(f"Mean count: {np.mean(counts_partB_train):.2f}")
print(f"Median count: {np.median(counts_partB_train):.2f}")
print(f"Max count: {np.max(counts_partB_train)}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(counts_partB_train, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Crowd Counts (Part B - Train)")
plt.xlabel("Number of People in Image")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


sizes = []
img_dir = os.path.join(partB_train, "images")
for img_name in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir, img_name))
    h, w = img.shape[:2]
    sizes.append((w, h))

sizes = np.array(sizes)
unique_sizes = np.unique(sizes, axis=0)
print("Unique image sizes:\n", unique_sizes)
print("Average size:", np.mean(sizes, axis=0))
