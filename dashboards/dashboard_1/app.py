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
# MODEL DOWNLOAD SETTINGS
# =========================
MODEL_PATH = "csrnet_finetuned.pth"

# YOUR GOOGLE DRIVE FILE ID
GDRIVE_FILE_ID = "19WxyQffyzOiQ3ABp-Ogd5lQ4hxINBz0v"


# =========================
# LOAD MODEL (WITH DOWNLOAD)
# =========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        st.info("Model file not found. Downloading from Google Drive...")
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
    EMAIL_PASSWORD = "rgvj gfww eusn txtg"   
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
st.title("Crowd Monitoring Dashboard")

model, device = load_model()

ALERT_THRESHOLD = st.slider("Alert Threshold", 5, 200, 10)

uploaded_file = st.file_uploader("Upload Crowd Video", type=["mp4", "avi"])

frame_placeholder = st.empty()
status_placeholder = st.empty()

if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

if "count_buffer" not in st.session_state:
    st.session_state.count_buffer = deque(maxlen=5)


# =========================
# VIDEO PROCESSING
# =========================
if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, device).to(device)

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

        if alert_active:
            status = "🚨 OVERCROWDING ALERT"
            color = (0, 0, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)

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

