import cv2
import streamlit as st

from model_core import run_density
from alert_utils import add_email, get_emails, send_alert

st.set_page_config(page_title="Smart Crowd Monitoring", layout="wide")
st.title("🚦 Smart Crowd Monitoring System")

# ------- session state -------
if "video_done" not in st.session_state:
    st.session_state.video_done = False
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# ------- sidebar -------
st.sidebar.header("📧 Alert Emails")
email = st.sidebar.text_input("Add email")
if st.sidebar.button("Save Email") and email:
    add_email(email)

threshold = st.sidebar.slider("Alert threshold", 1, 300, 10)

# ------- input mode -------
input_mode = st.radio("Select input source", ["Live Webcam", "Video File"])

# ------- display areas -------
col1, col2, col3 = st.columns(3)
orig_box = col1.empty()
dens_box = col2.empty()
over_box = col3.empty()

count_box = st.empty()
alert_box = st.empty()

# ================= VIDEO FILE MODE =================
if input_mode == "Video File":
    uploaded = st.file_uploader("Upload video", ["mp4", "avi", "mov"])

    if uploaded:
        st.session_state.video_done = False
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded.read())

    if uploaded and not st.session_state.video_done:
        cap = cv2.VideoCapture("temp_video.mp4", cv2.CAP_FFMPEG)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                st.session_state.video_done = True
                st.success("✅ Video processing completed")
                break

            o, d, ov, count = run_density(frame, is_webcam=False)

            orig_box.image(o, channels="BGR", caption="Original")
            dens_box.image(d, channels="BGR", caption="Density Map")
            over_box.image(ov, channels="BGR", caption="Overlay")

            count_box.metric("👥 Crowd Count", int(round(count)))

            if count > threshold:
                alert_box.error("⚠️ CROWD ALERT")
                for e in get_emails():
                    try:
                        send_alert(e, count)
                    except Exception:
                        st.warning("Email send failed (check SMTP)")
            else:
                alert_box.success("SAFE")

# ================= WEBCAM MODE =====================
if input_mode == "Live Webcam":
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("▶ Start Webcam"):
            st.session_state.webcam_running = True
    with col_b2:
        if st.button("⏹ Stop Webcam"):
            st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(1)

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            o, d, ov, count = run_density(frame, is_webcam=True)

            orig_box.image(o, channels="BGR", caption="Original")
            dens_box.image(d, channels="BGR", caption="Density Map")
            over_box.image(ov, channels="BGR", caption="Overlay")

            count_box.metric("👥 Crowd Count", int(round(count)))

            if count > threshold:
                alert_box.error("⚠️ CROWD ALERT")
                for e in get_emails():
                    try:
                        send_alert(e, count)
                    except Exception:
                        st.warning("Email send failed (check SMTP)")
            else:
                alert_box.success("SAFE")

        cap.release()
