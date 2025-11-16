import cv2
import numpy as np
import torch

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(img, target_size=(512, 512)):
    return cv2.resize(img, target_size)

def normalize_image(img):
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std

def to_tensor(img):
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()

def preprocess_pipeline(path, target_size=(512, 512)):
    img = load_image(path)
    img = resize_image(img, target_size)
    img = normalize_image(img)
    tensor = to_tensor(img)
    return tensor
