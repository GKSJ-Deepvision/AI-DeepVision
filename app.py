import streamlit as st
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ================= CONFIGURATION =================
VIDEO_PATH = "Videos/video.mp4"
SENDER_EMAIL = "nshree03112005@gmail.com"
EMAIL_PASSWORD = "jahvjlezrfnqhyzj"  # Gmail App Password
EMAIL_DB = "emails.csv"
EMAIL_COOLDOWN = 60  # seconds
# =================================================

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Crowd Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>👥 AI Crowd Monitoring Dashboard</h1>
<p style='text-align:center;'>Real-time crowd detection with smart alerting system</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
THRESHOLD = st.sidebar.slider("🚨 Crowd Threshold", 10, 200, 50)
video_file = st.sidebar.file_uploader("📹 Upload Video", type=["mp4", "avi"])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- EMAIL FUNCTIONS ----------------
def load_emails():
    try:
        df = pd.read_csv(EMAIL_DB)
        return df["email"].dropna().tolist()
    except:
        return []

def send_email_alert(count, level):
    recipients = load_emails()
    if not recipients:
        st.warning("⚠️ No emails found in database.")
        return

    subject = f"🚨 Crowd {level} Alert"
    body = f"""
Crowd Alert Level: {level}

Current Crowd Count: {count}
Threshold: {THRESHOLD}

Please take immediate action if required.
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        server.quit()
        st.success("📧 Alert email sent successfully!")
    except Exception as e:
        st.error(f"Email Error: {e}")

# ---------------- VIDEO SOURCE ----------------
if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    cap = cv2.VideoCapture("temp_video.mp4")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

frame_placeholder = st.empty()
metric1, metric2, metric3 = st.columns(3)
alert_box = st.empty()

last_email_time = 0

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)[0]
    person_count = 0

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    # ---------------- METRICS ----------------
    metric1.metric("👥 Current Count", person_count)
    metric2.metric("🚨 Threshold", THRESHOLD)
    metric3.metric("📊 Utilization", f"{int((person_count/THRESHOLD)*100)}%")

    progress = min(person_count / THRESHOLD, 1.0)
    st.progress(progress)

    current_time = time.time()

    # ---------------- ALERT LEVELS ----------------
    if person_count >= THRESHOLD:
        alert_box.error(f"🔴 CRITICAL ALERT! Count: {person_count}")
        if current_time - last_email_time > EMAIL_COOLDOWN:
            send_email_alert(person_count, "CRITICAL")
            last_email_time = current_time

    elif person_count >= 0.8 * THRESHOLD:
        alert_box.warning(f"🟡 WARNING! Crowd Rising: {person_count}")

    else:
        alert_box.success(f"🟢 SAFE ZONE: {person_count}")

cap.release()
st.info("✅ Video processing completed.")
