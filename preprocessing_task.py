import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Reads in BGR format
    if img is None:
        print("Error: Unable to load image:", image_path)
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    normalized = resized / 255.0
    return normalized