import cv2
import torch
import numpy as np
import streamlit as st
import time
import os

from csrnet_model import CSRNet
from alert import send_email   # 🔔 EMAIL ALERT MODULE

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DeepVision Crowd Monitor – Milestone 4",
    layout="wide"
)

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CONFIG =================
MODEL_PATH = "csrnet_epoch105.pth"
CROWD_THRESHOLD = 70
ALERT_EMAIL = "thenovavoyage00@gmail.com"   # 🔔 CHANGE THIS
SNAPSHOT_DIR = "snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= SESSION STATE =================
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CSRNet().to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

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
st.title("🧠 DeepVision Crowd Monitor ")

mode = st.radio("Select Input Mode", ["Webcam", "Upload Video"])

frame_box = st.image([])
count_box = st.empty()
alert_box = st.empty()

# ================= WEBCAM MODE =================
if mode == "Webcam":
    start = st.checkbox("▶ Start Webcam")

    if start:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        prev_time = time.time()

        while cap.isOpened() and start:
            ret, frame = cap.read()
            if not ret:
                break

            overlay, count = process_frame(frame)

            now = time.time()
            fps = 1 / max(1e-6, now - prev_time)
            prev_time = now

            frame_box.image(overlay, channels="BGR")
            count_box.metric("👥 Crowd Count", count)

            if count >= CROWD_THRESHOLD:
                alert_box.error("🚨 OVERCROWDING DETECTED")

                if not st.session_state.alert_sent:
                    snapshot_path = f"{SNAPSHOT_DIR}/alert_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, overlay)

                    send_email(
                        to_email=ALERT_EMAIL,
                        count=count,
                        alert_text="Overcrowding detected",
                        fps=fps,
                        snapshot_path=snapshot_path
                    )

                    st.session_state.alert_sent = True
            else:
                alert_box.success("✅ Crowd Level Normal")
                st.session_state.alert_sent = False

        cap.release()

# ================= VIDEO MODE =================
if mode == "Upload Video":
    video = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video:
        with open("temp.mp4", "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture("temp.mp4")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_time = time.time()
        frame_id = 0
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            overlay, count = process_frame(frame)

            now = time.time()
            fps = 1 / max(1e-6, now - prev_time)
            prev_time = now

            frame_box.image(overlay, channels="BGR")

            elapsed = time.time() - start_time
            progress = (frame_id / total) * 100 if total > 0 else 0

            count_box.markdown(
                f"""
🎬 **Frame:** {frame_id}/{total}  
📊 **Progress:** {progress:.1f}%  
👥 **Count:** {count}  
⚡ **FPS:** {fps:.1f}
"""
            )

            if count >= CROWD_THRESHOLD:
                alert_box.error("🚨 OVERCROWDING DETECTED")

                if not st.session_state.alert_sent:
                    snapshot_path = f"{SNAPSHOT_DIR}/alert_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, overlay)

                    send_email(
                        to_email=ALERT_EMAIL,
                        count=count,
                        alert_text="Overcrowding detected",
                        fps=fps,
                        snapshot_path=snapshot_path
                    )

                    st.session_state.alert_sent = True
            else:
                alert_box.success("✅ Crowd Level Normal")
                st.session_state.alert_sent = False

        cap.release()
        st.success("🎉 Video processed successfully!")
