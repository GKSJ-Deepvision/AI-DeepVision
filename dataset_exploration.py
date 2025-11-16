#worked
# lists the number of images in the various subfolders from the shanghaitech dataset
import os

dataset_path = r"C:\Users\mahal\OneDrive\Desktop\DL\archive\ShanghaiTech"

print("📌 Dataset Exploration\n")

# PART A
part_a_train_img = os.listdir(os.path.join(dataset_path, "part_A", "train_data", "images"))
part_a_train_gt  = os.listdir(os.path.join(dataset_path, "part_A", "train_data", "ground-truth"))

part_a_test_img = os.listdir(os.path.join(dataset_path, "part_A", "test_data", "images"))
part_a_test_gt  = os.listdir(os.path.join(dataset_path, "part_A", "test_data", "ground-truth"))

print("🔶 Part A")
print("Training images:", len(part_a_train_img))
print("Training ground-truth files:", len(part_a_train_gt))
print("Testing images:", len(part_a_test_img))
print("Testing ground-truth files:", len(part_a_test_gt), "\n")

# PART B
part_b_train_img = os.listdir(os.path.join(dataset_path, "part_B", "train_data", "images"))
part_b_train_gt  = os.listdir(os.path.join(dataset_path, "part_B", "train_data", "ground-truth"))

part_b_test_img = os.listdir(os.path.join(dataset_path, "part_B", "test_data", "images"))
part_b_test_gt  = os.listdir(os.path.join(dataset_path, "part_B", "test_data", "ground-truth"))

print("🔷 Part B")
print("Training images:", len(part_b_train_img))
print("Training ground-truth files:", len(part_b_train_gt))
print("Testing images:", len(part_b_test_img))
print("Testing ground-truth files:", len(part_b_test_gt))
