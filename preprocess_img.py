import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


# -----------------------------
# 1. Load a sample image
# -----------------------------
img_path = r"C:\Users\bolak\OneDrive\Desktop\deepvision\deepvision\ShanghaiTech\part_A\train_data\images\IMG_2.jpg"

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Failed to load image from {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")
plt.show()

# -----------------------------
# 2. Resize the image
# -----------------------------
resized_img = cv2.resize(img, (512, 512))

plt.figure(figsize=(6,6))
plt.title("Resized Image (512 × 512)")
plt.imshow(resized_img)
plt.axis("off")
plt.show()

# -----------------------------
# 3. Define preprocessing transforms
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),               # Convert to tensor [0–1]
    transforms.Normalize(                # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 4. Apply the transform
# -----------------------------
img_tensor = transform(resized_img)

print("Tensor Shape:", img_tensor.shape)
print("Min pixel value:", img_tensor.min().item())
print("Max pixel value:", img_tensor.max().item())