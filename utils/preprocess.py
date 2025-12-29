import cv2
import torch
import numpy as np

def preprocess_frame(frame):
    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize so dimensions are divisible by 8
    h, w, _ = frame.shape
    h = h - (h % 8)
    w = w - (w % 8)
    frame = cv2.resize(frame, (w, h))

    # Convert to float (keep float32 throughout to match model params)
    frame = frame.astype(np.float32) / np.float32(255.0)

    # ImageNet normalization (CRITICAL) - use float32 arrays to avoid upcasting
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std

    # Ensure still float32 before converting to tensor
    frame = frame.astype(np.float32)

    # HWC → CHW
    frame = frame.transpose(2, 0, 1)

    # Add batch dimension
    frame = torch.from_numpy(frame).unsqueeze(0)

    return frame
