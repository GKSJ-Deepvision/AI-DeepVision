import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found ->", image_path)
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    normalized = resized / 255.0

    return normalized

# Use your absolute image path here
image_path = r"C:\Users\Lenovo\Downloads\deep dataset\part_A_final\test_data\images\IMG_1.jpg"

output = preprocess_image(image_path)

if output is not None:
    print("Preprocessing successful!")