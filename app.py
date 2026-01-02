import cv2
import torch
import numpy as np
import streamlit as st
import os
import requests

from csrnet_model import CSRNet

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

# 🔗 GitHub Release Direct Download (IMPORTANT)
MODEL_URL = "https://github.com/GKSJ-Deepvision/AI-DeepVision/releases/download/v1.0/csrnet_epoch105.pth"

# ================= CONFIG =================
CROWD_THRESHOLD = 70

# ================= DOWNLOAD MODEL (SAFE METHOD) =================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    st.warning("⬇️ Downloading CSRNet model (one-time setup)...")

    response = requests.get(MODEL_URL, stream=True)
    if response.status_code != 200:
        st.error("❌ Failed to download model from GitHub Release")
        st.stop()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    st.success("✅ Model downloaded successfully")

download_model()

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner="🧠 Loading CSRNet model...")
def load_model():
    model = CSRNet().to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint

    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model

model = load_model()
st.success("✅ CSRNet model loaded successfully")

# ================= PREPROCESS =================
def preprocess(frame):
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std

    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)

# ================= PROCESS FRAME =================
def process_frame(frame):
    h, w, _ = frame.shape
    tensor = preprocess(frame)

    with torch.no_grad():
        density = model(tensor)

    density_map = density.squeeze().cpu().numpy()
    density_map[density_map < 0] = 0
    count = int(density_map.sum())

    heatmap = cv2.applyColorMap(
        (cv2.resize(density_map, (w, h)) * 255 / (density_map.max() + 1e-6)).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    return overlay, count

# ================= UI =================
st.title("🧠 DeepVision Crowd Monitor")
st.info("📌 Upload video to analyze crowd density")

video = st.file_uploader("Upload a video", type=["mp4", "avi"])

frame_box = st.image([])
count_box = st.empty()
alert_box = st.empty()

if video:
    temp_path = os.path.join(BASE_DIR, "temp.mp4")
    with open(temp_path, "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture(temp_path)

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
    st.success("🎉 Video processed successfully")
