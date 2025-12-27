import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from email_alert import send_email_alert
import time

VIDEO_PATH = "Videos/video.mp4"
THRESHOLD = 50
MODEL_PATH = "csrnet_weights.pth"

st.set_page_config(page_title="Crowd Density Monitoring Dashboard", layout="wide")
st.title("🧠 Crowd Density Monitoring Dashboard")

frame_area = st.empty()
status_area = st.empty()
count_area = st.empty()

# ---------------- MODEL CLASS ----------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # Example: replace with actual architecture
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.backend = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_weights():
    try:
        model = CSRNet()
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model_weights()
if model is None:
    st.stop()

cap = cv2.VideoCapture(VIDEO_PATH)
email_sent = False
running = st.button("▶ Start Monitoring")

while running and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_tensor = torch.tensor(gray/255.0).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        density_map = model(input_tensor)
        crowd_count = int(density_map.sum().item())

    density = density_map.squeeze().numpy()
    density = density / (density.max() + 1e-5)
    density = (density * 255).astype(np.uint8)
    density_colored = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, density_colored, 0.4, 0)

    frame_area.image(overlay, channels="BGR", use_container_width=True)
    count_area.info(f"👥 Crowd Count: {crowd_count}")

    if crowd_count > THRESHOLD:
        status_area.error("🚨 ALERT: High Crowd Density")
        if not email_sent:
            send_email_alert(crowd_count)
            email_sent = True
    else:
        status_area.success("✅ Status: Normal")
        email_sent = False

    time.sleep(0.3)

cap.release()
