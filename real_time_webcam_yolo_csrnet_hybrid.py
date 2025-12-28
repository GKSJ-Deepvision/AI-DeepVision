# ======================================================
# DeepVision – YOLO + CSRNet Hybrid Inference Module
# (Streamlit-safe, reusable, no execution on import)
# ======================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

# ======================================================
# Device
# ======================================================

DEVICE = torch.device("cpu")

# ======================================================
# CSRNet Architecture
# ======================================================

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)

        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

# ======================================================
# Load Models (ONCE)
# ======================================================

csrnet_model = CSRNet().to(DEVICE)
csrnet_model.load_state_dict(
    torch.load("csrnet_finetuned.pth", map_location=DEVICE)
)
csrnet_model.eval()
print("✔ CSRNet loaded")

yolo_model = YOLO("yolov8n.pt")
print("✔ YOLOv8 loaded")

# ======================================================
# Thresholds
# ======================================================

YOLO_THRESHOLD_LOW = 8
YOLO_THRESHOLD_HIGH = 20

# ======================================================
# Preprocess for CSRNet
# ======================================================

def preprocess_for_csrnet(frame_bgr):
    frame = cv2.resize(frame_bgr, (640, 360))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    return frame.to(DEVICE)

# ======================================================
# AUTO-SWITCH INFERENCE (USED BY STREAMLIT)
# ======================================================

def auto_switch_inference(frame_bgr):
    """
    Input:
        frame_bgr (OpenCV BGR frame)

    Output:
        vis_frame (BGR)
        crowd_count (int)
        mode (str)
    """

    vis_frame = frame_bgr.copy()

    # ---------- YOLO pass ----------
    results = yolo_model(frame_bgr, verbose=False)[0]
    boxes = results.boxes
    yolo_count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------- Decision ----------
    if yolo_count <= YOLO_THRESHOLD_LOW:
        mode = "YOLO"
        crowd_count = yolo_count

    else:
        with torch.no_grad():
            inp = preprocess_for_csrnet(frame_bgr)
            density = csrnet_model(inp)[0, 0]
            csr_count = int(density.sum().item())

        if yolo_count <= YOLO_THRESHOLD_HIGH:
            mode = "Hybrid"
            crowd_count = int((yolo_count + csr_count) / 2)
        else:
            mode = "CSRNet"
            crowd_count = csr_count

    # ---------- Overlay ----------
    cv2.putText(
        vis_frame,
        f"Count: {crowd_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        vis_frame,
        f"Mode: {mode}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

    return vis_frame, crowd_count, mode
