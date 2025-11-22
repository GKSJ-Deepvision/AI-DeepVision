import os
import cv2
import matplotlib.pyplot as plt


data_path = r"C:\Users\bolak\OneDrive\Desktop\deepvision\deepvision\ShanghaiTech\part_A\train_data\images"
images = os.listdir(data_path)
print(f"Total images found: {len(images)}")

# Display the first 4 sample images
plt.figure(figsize=(12, 8))
for i, img_name in enumerate(images[:4]):
    img_path = os.path.join(data_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Could not read {img_name}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(f"Sample {i + 1}: {img_name}")
    plt.axis("off")

plt.tight_layout()
plt.show()
