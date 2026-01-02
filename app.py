import cv2
import torch
import numpy as np
import streamlit as st
import time
import os
import urllib.request

from csrnet_model import CSRNet
from alert import send_email

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DeepVision Crowd Monitor – Milestone 4",
    layout="wide"
)

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "csrnet_epoch105.pth")

# 🔗 MODEL DOWNLOAD URL (YOUR FILE)
MODEL_URL = "https://github.com/GKSJ-Deepvision/AI-DeepVision/releases/download/v1.0/csrnet_epoch105.pth"

# ================= CONFIG =================
CROWD_THRESHOLD = 70
ALERT_EMAIL = "receiver@gmail.com"

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= ENSURE MODEL EXISTS =================
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("⬇️ Downloading CSRNet model (~53MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded")

ensure_model()

# ================= SESSION STATE =================
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CSRNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint

    fixed_state = {
        k.replace("module.", "").replace("core.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(fixed_state, strict=True)
    model.eval()
    return model

model = load_model()
st.success("✅ CSRNet model loaded successfully")

# ================= PREPROCESS =================
def preprocess(frame):
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std

    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)

# ================= PROCESS FRAME =================
def process_frame(frame):
    h, w, _ = frame.shape
    input_tensor = preprocess(frame)

    with torch.no_grad():
        density = model(input_tensor)

    density_map = density.squeeze().cpu().numpy()
    density_map[density_map < 0] = 0
    count = int(density_map.sum())

    density_vis = cv2.resize(density_map, (w, h))
    density_norm = density_vis / density_vis.max() if density_vis.max() > 0 else density_vis

    heatmap = cv2.applyColorMap(
        (density_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.75, heatmap, 0.25, 0)

    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(overlay, f"Count: {count}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    status = "CROWD ALERT" if count >= CROWD_THRESHOLD else "CROWD NORMAL"
    color = (0, 0, 255) if count >= CROWD_THRESHOLD else (0, 255, 0)

    cv2.putText(overlay, status, (260, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return overlay, count

# ================= UI =================
st.title("🧠 DeepVision Crowd Monitor")
st.info("📌 Webcam works locally | Upload Video works on Streamlit Cloud")

mode = st.radio("Select Input Mode", ["Upload Video"])

frame_box = st.image([])
count_box = st.empty()
alert_box = st.empty()

# ================= VIDEO MODE =================
video = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video:
    temp_video = os.path.join(BASE_DIR, "temp.mp4")
    with open(temp_video, "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture(temp_video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay, count = process_frame(frame)
        frame_box.image(overlay, channels="BGR")
        count_box.metric("👥 Crowd Count", count)

        if count >= CROWD_THRESHOLD:
            alert_box.error("🚨 OVERCROWDING DETECTED")
        else:
            alert_box.success("✅ Crowd Level Normal")

    cap.release()
    st.success("🎉 Video processed successfully!")
