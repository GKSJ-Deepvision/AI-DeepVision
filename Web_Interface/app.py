import streamlit as st
import cv2
import time
import pandas as pd
import os
from database import add_email, get_emails
from alert import send_email
from fusion import smart_infer

# ==================== PAGE CONFIG ====================
st.set_page_config("🚦 Smart Crowd Monitoring", layout="wide")
st.title("🚦 Smart Crowd Monitoring System")

# ==================== SESSION MEMORY ====================
if "running" not in st.session_state:
    st.session_state.running = False

if "last_frames" not in st.session_state:
    st.session_state.last_frames = None

if "df_webcam" not in st.session_state:
    st.session_state.df_webcam = pd.DataFrame(columns=["Count"])

if "df_video" not in st.session_state:
    st.session_state.df_video = pd.DataFrame(columns=["Count"])

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Webcam"

if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False  # Track alert email per alert

# ==================== STORAGE ====================
os.makedirs("reports", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

report_file = "reports/log.csv"
if not os.path.exists(report_file) or os.path.getsize(report_file) == 0:
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Time,Count,Alert,FPS\n")

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Control Panel")

input_mode = st.sidebar.selectbox("Input Source", ["Video File", "Webcam"])
threshold = st.sidebar.slider("Alert Threshold", 5, 300, 100)

if st.session_state.current_mode != input_mode:
    st.session_state.current_mode = input_mode

with st.sidebar.expander("📧 Alert Emails"):
    email = st.text_input("Add Email")
    if st.button("Save Email"):
        if email:
            add_email(email)
    for e in get_emails():
        st.write("•", e)

if input_mode == "Video File":
    uploaded = st.sidebar.file_uploader("Upload Video", ["mp4","avi"])
    video_uploaded = uploaded is not None
else:
    video_uploaded = False

start_col, stop_col = st.sidebar.columns(2)
start_btn = start_col.button("▶ Start")
stop_btn = stop_col.button("⏹ Stop")

# ==================== UI PLACEHOLDERS ====================
# Top message for start/stop alerts
status_msg = st.empty()

# Frames placeholders
cols = st.columns(3)
frame_original = cols[0].empty()
frame_density  = cols[1].empty()
frame_overlay  = cols[2].empty()

count_box = st.metric("👥 Crowd Count", "0")
alert_box = st.empty()
chart_box = st.empty()

# ==================== START / STOP ====================
if start_btn:
    if input_mode == "Webcam":
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.running = True
        st.session_state.alert_sent = False
        status_msg.success("✅ Monitoring started with Webcam")
    elif video_uploaded:
        with open("temp.mp4","wb") as f:
            f.write(uploaded.read())
        st.session_state.cap = cv2.VideoCapture("temp.mp4")
        st.session_state.running = True
        st.session_state.alert_sent = False
        status_msg.success("✅ Monitoring started with Video File")
    else:
        status_msg.warning("Upload a video first")

if stop_btn:
    st.session_state.running = False
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap
    status_msg.info("🛑 Monitoring stopped")

# ==================== MAIN LOOP ====================
if "cap" in st.session_state and st.session_state.running:
    cap = st.session_state.cap
    prev_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        use_yolo = input_mode == "Webcam"
        original, density_map, overlay, count = smart_infer(frame, use_yolo)

        # FPS calculation
        now = time.time()
        fps = 1 / max(1e-6, now - prev_time)
        prev_time = now

        # Alert logic
        alert_text = "SAFE"
        color = "green"
        snapshot_path = None

        if count > threshold:
            alert_text = "⚠️ CROWD ALERT"
            color = "red"

            # Save snapshot only on alert
            snapshot_path = f"snapshots/alert_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_path, overlay)

            # Send emails only once per alert
            if not st.session_state.alert_sent:
                for e in get_emails():
                    send_email(
                        to_email=e,
                        count=int(count),
                        alert_text=alert_text,
                        fps=fps,
                        snapshot_path=snapshot_path
                    )
                status_msg.success(f"🚨 Alert email sent! Count: {int(count)}, FPS: {fps:.1f}")
                st.session_state.alert_sent = True
        else:
            st.session_state.alert_sent = False  # Reset when count goes below threshold

        # UI update
        count_box.metric("👥 Crowd Count", f"{count:.1f}")
        alert_box.markdown(
            f"<h3 style='color:{color}'>{alert_text} | FPS {fps:.1f}</h3>",
            unsafe_allow_html=True
        )

        # Save log
        with open(report_file,"a", encoding="utf-8") as f:
            f.write(f"{time.ctime()},{int(count)},{alert_text},{fps:.2f}\n")

        # Store history separately
        if input_mode == "Webcam":
            st.session_state.df_webcam = pd.concat(
                [st.session_state.df_webcam, pd.DataFrame({"Count":[count]})],
                ignore_index=True
            )
            active_df = st.session_state.df_webcam
        else:
            st.session_state.df_video = pd.concat(
                [st.session_state.df_video, pd.DataFrame({"Count":[count]})],
                ignore_index=True
            )
            active_df = st.session_state.df_video

        # Display frames larger
        frame_original.image(cv2.cvtColor(original,cv2.COLOR_BGR2RGB), caption="Original", width=420)
        frame_density.image(cv2.cvtColor(density_map,cv2.COLOR_BGR2RGB), caption="Density", width=420)
        frame_overlay.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB), caption="Overlay", width=420)

        chart_box.line_chart(active_df["Count"])

        st.session_state.last_frames = (original, density_map, overlay)

    cap.release()

# ==================== PERSIST DISPLAY AFTER STOP ====================
if not st.session_state.running and st.session_state.last_frames is not None:
    o, d, ov = st.session_state.last_frames
    frame_original.image(cv2.cvtColor(o,cv2.COLOR_BGR2RGB), width=720)
    frame_density.image(cv2.cvtColor(d,cv2.COLOR_BGR2RGB), width=720)
    frame_overlay.image(cv2.cvtColor(ov,cv2.COLOR_BGR2RGB), width=720)

    active_df = st.session_state.df_webcam if input_mode == "Webcam" else st.session_state.df_video
    chart_box.line_chart(active_df["Count"])

# ==================== DOWNLOAD ====================
active_df = st.session_state.df_webcam if input_mode == "Webcam" else st.session_state.df_video
if not active_df.empty:
    st.download_button(
        "⬇ Download Report",
        open(report_file,"rb"),
        "log.csv"
    )
