import os
from PIL import Image
import matplotlib.pyplot as plt

# --- STEP 1: Path to your dataset folder ---
folder_path = r"C:\Users\nshre\OneDrive\Desktop\NITHYA-SHREE\Crowd"

# --- STEP 2: Get all image files ---
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
          if os.path.splitext(f)[1].lower() in image_extensions]

print(f"✅ Found {len(images)} images in '{folder_path}'")

# --- STEP 3: Display sample images ---
if len(images) == 0:
    print("No images found. Please check the folder path or file types.")
else:
    plt.figure(figsize=(10, 6))
    for i, img_path in enumerate(images[:9]):  # show first 9 images
        img = Image.open(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(img_path)[:15])
    plt.tight_layout()
    plt.show()
