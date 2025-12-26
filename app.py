import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import smtplib
from email.mime.text import MIMEText

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Crowd Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== SMTP CONFIG ==================
SENDER_EMAIL = "srichandanachavali@gmail.com"   # FIXED sender
APP_PASSWORD = "YOUR_GMAIL_APP_PASSWORD"        # MUST be Gmail App Password

def send_alert_email(receiver_email, count):
    try:
        msg = MIMEText(
            f"⚠️ Overcrowding Detected!\n\nCrowd Count: {count}\n\nPlease take necessary action."
        )
        msg["Subject"] = "Crowd Alert Notification"
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        return True, "Email sent successfully"

    except Exception as e:
        return False, str(e)

# ================== CSRNet ==================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(512,256,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(256,128,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(128,64,3,padding=2,dilation=2), nn.ReLU(),
            nn.Conv2d(64,1,1)
        )

    def forward(self, x):
        return self.backend(self.frontend(x))

# ================== MODEL & TRANSFORM ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================== HEADER ==================
st.markdown("## 🚦 AI Crowd Monitoring Dashboard")
st.caption("Real-time crowd density estimation, alerting, and monitoring")

# ================== SIDEBAR ==================
st.sidebar.markdown("### ⚙️ Alert Configuration")

threshold = st.sidebar.slider(
    "Crowd Threshold",
    min_value=50,
    max_value=500,
    value=150
)

receiver_email = st.sidebar.text_input(
    "Send alert email to",
    placeholder="example@gmail.com"
)

test_email_btn = st.sidebar.button("📧 Send Test Email")

if test_email_btn:
    if receiver_email:
        ok, msg = send_alert_email(receiver_email, count=0)
        if ok:
            st.sidebar.success("✅ Test email sent")
        else:
            st.sidebar.error(f"❌ Email failed: {msg}")
    else:
        st.sidebar.warning("⚠️ Enter an email address")

# ================== MAIN UI ==================
st.markdown("### 📹 Video Input")
video_file = st.file_uploader(
    "Upload CCTV-style video",
    type=["mp4", "avi"],
    help="Upload a crowd video for analysis"
)

frame_col, stats_col = st.columns([3, 1])

email_sent = False

if video_file:
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        frame_rs = cv2.resize(frame_rgb, (w//8*8, h//8*8))

        img = transform(frame_rs).unsqueeze(0).to(device)

        with torch.no_grad():
            density = model(img)[0, 0].cpu().numpy()
            count = max(0, int(density.sum()))  # avoid negative counts

        # -------- DISPLAY --------
        with frame_col:
            st.image(frame_rgb, channels="RGB", use_container_width=True)

        with stats_col:
            st.metric("👥 Crowd Count", count)
            st.metric("🚨 Threshold", threshold)

            if count > threshold:
                st.error("⚠️ OVERCROWDING DETECTED")

                if receiver_email and not email_sent:
                    ok, msg = send_alert_email(receiver_email, count)
                    if ok:
                        st.toast("📧 Alert email sent")
                    else:
                        st.toast(f"❌ Email failed: {msg}")
                    email_sent = True
            else:
                st.success("✅ Crowd Level Normal")
                email_sent = False

    cap.release()

else:
    st.info("⬆️ Upload a video to start monitoring")

# ================== FOOTER ==================
st.markdown("---")
st.caption("© AI DeepVision | CSRNet-based Crowd Monitoring System")
