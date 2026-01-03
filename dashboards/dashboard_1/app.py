import streamlit as st
import cv2
import torch
import numpy as np
from collections import deque
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
import os
import gdown

from csrnet_model import CSRNet
from utils import preprocess_frame, density_to_heatmap


# =========================
# MODEL SETTINGS
# =========================
MODEL_PATH = "csrnet_finetuned.pth"
GDRIVE_FILE_ID = "19WxyQffyzOiQ3ABp-Ogd5lQ4hxINBz0v"


# =========================
# LOAD MODEL (AUTO DOWNLOAD)
# =========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading CSRNet model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = CSRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# =========================
# EMAIL ALERT FUNCTION
# =========================
def send_email_alert(count):
    EMAIL_SENDER = "s91596532@gmail.com"
    EMAIL_PASSWORD = "rgvjgfww eusntxtg".replace(" ", "")  # App password
    EMAIL_RECEIVER = "kandesrilakshmi764@gmail.com"

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = "⚠️ Crowd Alert Detected"

    msg.set_content(f"""
ALERT: Overcrowding Detected

Crowd Count : {count}
Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CSRNet Crowd Monitoring System
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Crowd Monitoring Dashboard", layout="centered")

st.title("Crowd Monitoring Dashboard")
st.markdown("CSRNet-based crowd density estimation with alerts")

model, device = load_model()

ALERT_THRESHOLD = st.slider("Alert Threshold", 5, 200, 10)

mode = st.radio(
    "Select Input Mode",
    ["Upload Video", "Live Webcam"]
)

frame_placeholder = st.empty()
status_placeholder = st.empty()


# =========================
# SESSION STATE
# =========================
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

if "count_buffer" not in st.session_state:
    st.session_state.count_buffer = deque(maxlen=5)


# =========================
# VIDEO UPLOAD MODE
# =========================
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Crowd Video", type=["mp4", "avi"])

    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess_frame(frame, device)

            with torch.no_grad():
                density = model(input_tensor)

            count = density.sum().item()

            st.session_state.count_buffer.append(count)
            smooth_count = int(
                sum(st.session_state.count_buffer) /
                len(st.session_state.count_buffer)
            )

            heatmap = density_to_heatmap(density, frame.shape)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            alert_active = smooth_count >= ALERT_THRESHOLD

            color = (0, 0, 255) if alert_active else (0, 255, 0)
            status = "🚨 OVERCROWDING ALERT" if alert_active else "NORMAL"

            cv2.putText(
                overlay,
                f"Count: {smooth_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            frame_placeholder.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )
            status_placeholder.subheader(status)

            if alert_active and not st.session_state.email_sent:
                send_email_alert(smooth_count)
                st.session_state.email_sent = True

            if not alert_active:
                st.session_state.email_sent = False

            time.sleep(0.03)

        cap.release()


# =========================
# WEBCAM SNAPSHOT MODE
# =========================
if mode == "Live Webcam":
    st.subheader("Webcam Snapshot Crowd Monitoring")

    st.info(
        "Click 'Take Photo' to capture a frame."
    )

    camera_frame = st.camera_input("Take Photo")

    if camera_frame is not None:
        # Convert to OpenCV image
        img_bytes = camera_frame.getvalue()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # -------------------------
        # CSRNet INFERENCE
        # -------------------------
        input_tensor = preprocess_frame(frame, device)

        with torch.no_grad():
            density = model(input_tensor)

        count = density.sum().item()

        # Smooth count (same as video)
        st.session_state.count_buffer.append(count)
        smooth_count = int(
            sum(st.session_state.count_buffer) /
            len(st.session_state.count_buffer)
        )

        # -------------------------
        # HEATMAP OVERLAY
        # -------------------------
        heatmap = density_to_heatmap(density, frame.shape)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        alert_active = smooth_count >= ALERT_THRESHOLD

        color = (0, 0, 255) if alert_active else (0, 255, 0)
        status = "🚨 OVERCROWDING ALERT" if alert_active else "NORMAL"

        cv2.putText(
            overlay,
            f"Count: {smooth_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # -------------------------
        # DISPLAY RESULT
        # -------------------------
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption=status
        )

        # -------------------------
        # EMAIL ALERT
        # -------------------------
        if alert_active and not st.session_state.email_sent:
            send_email_alert(smooth_count)
            st.session_state.email_sent = True

        if not alert_active:
            st.session_state.email_sent = False

