import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------- PATHS -----------
csv_path = r"E:\DeepVision\crowds_counting.csv"
image_base_path = r"E:\DeepVision\ShanghaiTech\part_A\test_data\images"

# ----------- LOAD DATASET -----------
df = pd.read_csv(csv_path)
print("✅ Dataset Loaded Successfully")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# ----------- VISUALIZE IMAGES FROM SHANGHAITECH -----------
def visualize_shanghaitech_images(base_path, num_images=482):
    """Visualize images from ShanghaiTech dataset"""
    img_files = [f for f in os.listdir(base_path) if f.endswith('.jpg')]
    img_files = sorted(img_files)[:num_images]
    
    if len(img_files) == 0:
        print(f"❌ No images found in {base_path}")
        return
    
    plt.figure(figsize=(15, 3))
    
    for idx, img_name in enumerate(img_files, 1):
        img_path = os.path.join(base_path, img_name)
        
        try:
            img = Image.open(img_path)
            
            plt.subplot(1, len(img_files), idx)
            plt.imshow(img)
            plt.title(f"{img_name}\nSize: {img.size}")
            plt.axis("off")
            print(f"✅ Loaded: {img_name}")
        except Exception as e:
            print(f"❌ Error loading {img_name}: {e}")
    
    plt.tight_layout()
    plt.show()


# ----------- CALL FUNCTION -----------
print(f"\n📊 Visualizing images from: {image_base_path}")
visualize_shanghaitech_images(image_base_path, num_images=5)
