import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from csrnet_model import CustomCSRNet
from yolo_people_counter import count_people_yolo

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- LOAD CSRNET ----------------
csrnet = CustomCSRNet().to(device)
csrnet.load_state_dict(torch.load("best_finetuned.pth", map_location=device))
csrnet.eval()

# ---------------- IMAGENET NORMALIZATION ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- CALIBRATION ----------------
SPARSE_CALIBRATION = 0.2   # for sparse scenes

# ---------------- MAIN FUNCTION ----------------
def run_density(frame, is_webcam=False):
    """
    Robust fusion-based crowd estimation:
    - CSRNet for density patterns
    - YOLO for sanity & sparse correction
    """

    # ---------- PREPROCESS ----------
    resized = cv2.resize(frame, (640, 480))
    img = transform(resized).unsqueeze(0).to(device)

    # ---------- CSRNET INFERENCE ----------
    with torch.no_grad():
        density = csrnet(img)[0, 0].cpu().numpy()

    # ---------- STEP 1: COMPUTE BOTH ----------
    csr_count = float(density.sum())
    yolo_count = count_people_yolo(frame)

    # ---------- STEP 2: ADAPTIVE FUSION ----------
    if csr_count < 15:
        # Sparse → YOLO dominates
        final_count = max(csr_count * SPARSE_CALIBRATION, yolo_count)

    elif csr_count < 60:
        # Medium → blend CSRNet + YOLO
        final_count = 0.5 * (csr_count * SPARSE_CALIBRATION) + 0.5 * yolo_count

    else:
        # Dense → trust CSRNet trend, clamp with YOLO
        final_count = min(csr_count, yolo_count * 5)

    # ---------- SAFETY ----------
    final_count = max(0, final_count)

    # ==================================================
    # VISUALIZATION 
    # ==================================================

    # ---------- SMOOTHING ----------
    if is_webcam:
        density_vis = cv2.GaussianBlur(density, (31, 31), 0)
    else:
        density_vis = cv2.GaussianBlur(density, (15, 15), 0)

    # ---------- NORMALIZATION ----------
    max_val = density_vis.max()
    if max_val > 0:
        density_norm = density_vis / max_val
    else:
        density_norm = density_vis

    density_norm = np.clip(density_norm, 0, 1)

    # ---------- COLOR MAP ----------
    heatmap = np.uint8(255 * density_norm)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(
        heatmap,
        (frame.shape[1], frame.shape[0])
    )

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return frame, heatmap, overlay, final_count
