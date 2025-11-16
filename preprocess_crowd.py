import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# --- STEP 1: Path to your dataset folder ---
dataset_path = r"C:\Users\nshre\OneDrive\Desktop\NITHYA-SHREE\Crowd"

# --- STEP 2: Path to save preprocessed images ---
output_folder = r"C:\Users\nshre\OneDrive\Desktop\NITHYA-SHREE\Crowd_Preprocessed"
os.makedirs(output_folder, exist_ok=True)

# --- STEP 3: Define preprocessing steps ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(            # Normalize pixel values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- STEP 4: Get all image files ---
image_extensions = {'.jpg', '.jpeg', '.png'}
image_paths = [os.path.join(dataset_path, f)
               for f in os.listdir(dataset_path)
               if os.path.splitext(f)[1].lower() in image_extensions]

if not image_paths:
    print("❌ No images found! Please check your dataset path.")
else:
    print(f"✅ Found {len(image_paths)} images in '{dataset_path}'")

# --- STEP 5: Preprocess and save images ---
for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")     # Open and convert to RGB
    image_resized = image.resize((224, 224))        # Resize image

    # Save the resized image to the output folder
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    image_resized.save(save_path)

print(f"\n✅ Saved preprocessed images to: {output_folder}")

# --- STEP 6: Visualize a few preprocessed samples ---
sample_images = os.listdir(output_folder)[:9]

if sample_images:
    plt.figure(figsize=(10, 6))
    for i, filename in enumerate(sample_images):
        img_path = os.path.join(output_folder, filename)
        img = Image.open(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(filename[:15])
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No images found to visualize.")

