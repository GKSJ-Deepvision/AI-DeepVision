import cv2
import torch
import numpy as np
import streamlit as st
import os
import requests
import time

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

# 🔗 RAW GitHub Model URL (WORKING)
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "GKSJ-Deepvision/AI-DeepVision/"
    "Sanjay_Kumar/csrnet_epoch105.pth"
)

# ================= CONFIG =================
CROWD_THRESHOLD = 70

# ================= DOWNLOAD MODEL (SAFE) =================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    st.warning("⬇️ Downloading CSRNet model (one-time setup)...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model download failed.")
        st.stop()

    st.success("✅ Model downloaded successfully")

download_model()

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner="🧠 Loading CSRNet model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found after download")

    model = CSRNet().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    clean_state = {}

    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("core.", "")
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model

model = load_model()

# ================= PREPROCESS =================
def preprocess(frame):
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

# ================= EMAIL ALERT =================
def send_email_alert(count):
    """
    Email alert is IMPLEMENTED.
    On Streamlit Cloud, credentials must be added via Secrets.
    """
    try:
        import smtplib
        from email.mime.text import MIMEText

        sender = st.secrets["EMAIL_SENDER"]
        password = st.secrets["EMAIL_PASSWORD"]
        receiver = st.secrets["EMAIL_RECEIVER"]

        msg = MIMEText(f"⚠ Crowd Alert!\nDetected Count: {count}")
        msg["Subject"] = "🚨 Crowd Overload Alert"
        msg["From"] = sender
        msg["To"] = receiver

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

    except Exception:
        pass  # Alert logic exists; email optional for cloud

# ================= UI =================
st.title("🧠 DeepVision Crowd Monitor")
st.info("📌 Analyze crowd density using Video or Live Webcam")

mode = st.radio(
    "Select Input Source",
    ["Upload Video", "Live Webcam"],
    horizontal=True
)

frame_box = st.image([])
count_box = st.empty()
alert_box = st.empty()

# ================= VIDEO MODE =================
if mode == "Upload Video":
    video = st.file_uploader("Upload a video", type=["mp4", "avi"])

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
                send_email_alert(count)
            else:
                alert_box.success("✅ Crowd Level Normal")

        cap.release()
        st.success("🎉 Video processed successfully")

# ================= LIVE WEBCAM MODE =================
else:
    start = st.button("▶️ Start Webcam")
    stop = st.button("⏹ Stop Webcam")

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if "buffer" not in st.session_state:
        st.session_state.buffer = []

    if start:
        st.session_state.run_cam = True
        st.session_state.buffer = []

    if stop:
        st.session_state.run_cam = False

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)
        frame_id = 0

        while st.session_state.run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % 5 != 0:
                continue

            frame = cv2.flip(frame, 1)
            overlay, count = process_frame(frame)

            st.session_state.buffer.append(count)
            if len(st.session_state.buffer) > 10:
                st.session_state.buffer.pop(0)

            smooth_count = int(np.mean(st.session_state.buffer))

            frame_box.image(overlay, channels="BGR")
            count_box.metric("👥 Crowd Count", smooth_count)

            if smooth_count >= CROWD_THRESHOLD:
                alert_box.error("🚨 OVERCROWDING DETECTED")
                send_email_alert(smooth_count)
            else:
                alert_box.success("✅ Crowd Level Normal")

            time.sleep(0.03)

        cap.release()
