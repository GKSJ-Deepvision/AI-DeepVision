# explore_dataset.py
import os
from glob import glob
from dataset.visualization import visualize_sample, plot_count_histogram
from dataset.utils import list_image_gt_pairs
from config import BASE_DATASET_PATH as DATASET_ROOT


def explore():
    parts = ["part_A", "part_B"]
    splits = ["train_data", "test_data"]

    for part in parts:
        for split in splits:
            root = os.path.join(DATASET_ROOT, part, split)
            img_paths, gt_paths = list_image_gt_pairs(root)
            print(f"{part}/{split} -> images: {len(img_paths)}, gts: {len(gt_paths)}")

            if len(gt_paths) > 0 and split == "train_data":
                plot_count_histogram(gt_paths, title=f"{part} - Train Count Histogram")

            if len(img_paths) > 0:
                print("Showing one sample...")
                visualize_sample(img_paths[0], gt_paths[0])


if __name__ == "__main__":
    explore()
