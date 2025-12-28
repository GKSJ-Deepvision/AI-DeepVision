import streamlit as st
import cv2
import time

from real_time_webcam_yolo_csrnet_hybrid import auto_switch_inference
from alert_system import send_email_alert

st.set_page_config(layout="wide")
st.title("📊 DeepVision Crowd Monitoring Dashboard")
# Initialize session state for email alerts
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False


# UI placeholders
video_col, info_col = st.columns([3, 1])
frame_placeholder = video_col.empty()

count_metric = info_col.metric("👥 Crowd Count", "0")
mode_text = info_col.markdown("### 🧠 Mode: -")
alert_text = info_col.markdown("### 🚨 Alert: NO")

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Alert Threshold", 1, 50, 10)
start = st.sidebar.button("▶ Start Monitoring")
stop = st.sidebar.button("⏹ Stop")

# Webcam
cap = cv2.VideoCapture(0)

if start:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        vis, count, mode = auto_switch_inference(frame)
        alert = count > threshold
        # --------------------------------------------------
        # Alert + Email Logic
        # --------------------------------------------------
        if alert and not st.session_state.email_sent:
            send_email_alert(count, mode)
            st.session_state.email_sent = True

        if not alert:
            st.session_state.email_sent = False


        # Update UI
        frame_placeholder.image(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        count_metric.metric("👥 Crowd Count", int(count))
        mode_text.markdown(f"### 🧠 Mode: **{mode}**")
        alert_text.markdown(
            f"### 🚨 Alert: {'🔴 YES' if alert else '🟢 NO'}"
        )

        if stop:
            break

        time.sleep(0.03)

cap.release()
