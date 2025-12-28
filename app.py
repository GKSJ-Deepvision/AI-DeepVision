import cv2
import torch
import streamlit as st
import numpy as np

from csrnet_model import CSRNet
from utils_email import init_db, get_all_emails, send_alert_emails

# ================= CONFIG =================
st.set_page_config(page_title="Crowd Monitoring System (CSRNet)", layout="wide")

DEVICE = "cpu"   # CSRNet is stable on CPU
MODEL_PATH = r"C:\Users\Shiva Teja\Desktop\DeepVision\models\csrnet_video_finetuned_final.pth"

DENSITY_ALERT_THRESHOLD = 70.0   # Crowd Density Index threshold

init_db()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CSRNet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model = load_model()

# ================= PROCESS SINGLE FRAME =================
def process_frame(frame):
    H, W, _ = frame.shape

    img = cv2.resize(frame, (640, 360))
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization (CSRNet requirement)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img_t = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    with torch.no_grad():
        density = torch.relu(model(img_t))

    density_map = density.squeeze().cpu().numpy()

    # ---- Crowd Density Index ----
    density_index = float(density_map.sum())

    # ---- Density Map Visualization ----
    density_map = cv2.GaussianBlur(density_map, (13, 13), 0)
    density_map = cv2.resize(density_map, (W, H))

    p98 = np.percentile(density_map, 98)
    density_vis = np.clip(density_map / (p98 + 1e-6), 0, 1)

    heatmap = cv2.applyColorMap(
        (density_vis * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    return overlay, density_index

# ================= VIDEO LEVEL ESTIMATION =================
def estimate_from_video(video_path, n_frames=10):
    cap = cv2.VideoCapture(video_path)

    density_values = []
    last_overlay = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // n_frames, 1)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_step == 0:
            overlay, di = process_frame(frame)
            density_values.append(di)
            last_overlay = overlay

        idx += 1
        if len(density_values) >= n_frames:
            break

    cap.release()

    if not density_values:
        return None, 0.0

    avg_density_index = float(np.mean(density_values))
    return last_overlay, avg_density_index

# ================= UI =================
st.title("🧠 Crowd Monitoring System (CSRNet)")

uploaded_file = st.file_uploader(
    "Upload a crowd video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    overlay, density_index = estimate_from_video("temp.mp4")

    if overlay is not None:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(overlay, caption="Crowd Density Map", width=700)

        with col2:
            st.metric("📊 Crowd Density Index", f"{density_index:.2f}")

            if density_index >= DENSITY_ALERT_THRESHOLD:
                st.error("⚠️ CROWD DENSITY ALERT")
                send_alert_emails(get_all_emails(), density_index)
                msg = send_alert_emails(get_all_emails(), density_index)
                st.warning(msg)
            else:
                st.success("✅ SAFE")
