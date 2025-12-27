import cv2
import numpy as np
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_frame(frame, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    return img

def density_to_heatmap(density, frame_shape):
    density = density.squeeze().cpu().numpy()
    density = density / (density.max() + 1e-6)
    density = (density * 255).astype(np.uint8)

    heatmap = cv2.resize(density, (frame_shape[1], frame_shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap
