import torch
import cv2
import numpy as np
from model import CSRNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = (640, 480)

# ---------------- LOAD MODEL ----------------
@torch.no_grad()
def load_model():
    model = CSRNet()
    ckpt = torch.load(r"D:\Python Projects\DeepVision\csrnet_mall_epoch_5.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def preprocess_frame(frame):
    frame = cv2.resize(frame, TARGET)
    img = frame / 255.0
    img = img.transpose(2,0,1)
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(DEVICE)
    return img
def csrnet_infer(frame):
    inp = preprocess_frame(frame)
    with torch.no_grad():
        density = model(inp)[0,0].cpu().numpy()
    
    count = density.sum()

    # Density Heatmap
    heatmap = density / (density.max() + 1e-6)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # Return original frame, density heatmap, overlay, and count
    return frame, heatmap, overlay, count
