import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- DEVICE ----------------
device = torch.device("cpu")   # change to "cuda" if you want GPU and it works

# ---------------- CSRNET MODEL ----------------
class CustomCSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super().__init__()
        vgg = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if load_pretrained else None
        )
        # same cutoff as in your notebook
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
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
        )
        self.output_layer = nn.Sequential(
    nn.Conv2d(128, 1, 1),
    nn.ReLU(inplace=True)
)


    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ---------------- LOAD CSRNET WEIGHTS ----------------
csrnet = CustomCSRNet(load_pretrained=False).to(device)
state = torch.load("csrnet_finetuned_mall.pth", map_location=device)
csrnet.load_state_dict(state, strict=True)
csrnet.eval()

# ---------------- LOAD YOLO ----------------
yolo_model = YOLO("yolov8n.pt")  # or your own YOLO weights

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

SPARSE_CALIBRATION = 0.1

def count_people_yolo(frame):
    """Count persons in a BGR frame using YOLO."""
    results = yolo_model.predict(frame, verbose=False)
    total = 0
    for r in results:
        if r.boxes is not None:
            total += (r.boxes.cls == 0).sum().item()
    return int(total)
def run_density(frame, is_webcam=False, scene_type="street"):
    """
    Webcam : YOLO only (as before)
    Video  : CSRNet + per-scene scaling
             scene_type = "street" or "mall"
    """

    h, w = frame.shape[:2]
    clean_frame = frame.copy()

    # ================== WEBCAM MODE ==================
    if is_webcam:
        conf_thresh = 0.5
        iou_thresh = 0.6

        results = yolo_model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        yolo_count = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > conf_thresh:
                yolo_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(clean_frame, (x1, y1), (x2, y2),
                              (0, 180, 180), 2)
                cv2.putText(clean_frame, f"{conf:.2f}",
                            (x1, max(10, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        final_count = int(yolo_count)

    # ================== VIDEO MODE ==================
    else:
        # 1) CSRNet forward
        inp = cv2.resize(frame, (512, 512))
        inp = inp / 255.0
        inp = inp.transpose(2, 0, 1)
        inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            density_map = csrnet(inp).detach().cpu().numpy()[0, 0]

        # 2) Resize to original
        density_map = cv2.resize(density_map, (w, h))

        # 3) Raw sum
        raw_sum = float(density_map.sum())

        # 4) Calibrated scale per scene type
        if scene_type == "street":
            # For street crowd video (Istiklal): ~100–150 people
            scale_factor = 400.0   # you can tweak 350–500
        elif scene_type == "mall":
            # For mall / escalator videos: ~30–80 people
            scale_factor = 800.0   # you can tweak 700–900
        else:
            # fallback
            scale_factor = 600.0

        final_count = int(raw_sum / scale_factor)

        # 5) Draw density count
        cv2.rectangle(clean_frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(clean_frame, f"DENSITY COUNT: {final_count}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2, cv2.LINE_AA)

    # ================== ALERT TEXT ==================
    if final_count <= 5:
        alert_text = "LOW CROWD"
        alert_color = (80, 220, 120)
    elif final_count <= 20:
        alert_text = "SAFE"
        alert_color = (0, 200, 255)
    else:
        alert_text = "CROWD ALERT"
        alert_color = (0, 0, 255)

    cv2.putText(clean_frame, alert_text,
                (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                alert_color, 2, cv2.LINE_AA)

    # ================== HEATMAP (CSRNet) ==================
    inp_hm = cv2.resize(frame, (512, 512))
    inp_hm = inp_hm / 255.0
    inp_hm = inp_hm.transpose(2, 0, 1)
    inp_hm = torch.tensor(inp_hm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        density_hm = csrnet(inp_hm).detach().cpu().numpy()[0, 0]

    density_vis = cv2.resize(density_hm, (w, h))
    if density_vis.max() > 0:
        density_vis = density_vis / density_vis.max()
    density_vis = np.clip(density_vis, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * density_vis), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    return clean_frame, heatmap, overlay, final_count
