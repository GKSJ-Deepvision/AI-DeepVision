import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import yagmail
from torchvision import transforms

# =========================
# CONFIG
# =========================
SWITCH_THRESHOLD = 75
ALERT_THRESHOLD = 120

MODEL_A_PATH = "csrnet_final.pth"
MODEL_B_PATH = "csrnet_finalB.pth"

ALERT_EMAIL = "ishita22.d@gmail.com"   

# =========================
# CSRNet MODEL
# =========================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_A = CSRNet().to(device)
    model_B = CSRNet().to(device)

    model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
    model_B.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))

    model_A.eval()
    model_B.eval()

    return model_A, model_B, device

# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0)
    return tensor.to(device)

# =========================
# INFERENCE SWITCH
# =========================
def infer(frame_tensor, model_A, model_B):
    with torch.no_grad():
        dA = model_A(frame_tensor)
        countA = dA.sum().item()
        if countA > SWITCH_THRESHOLD:
            return dA, countA, "CSRNet Part A"
        dB = model_B(frame_tensor)
        return dB, dB.sum().item(), "CSRNet Part B"

# =========================
# HEATMAP
# =========================
def generate_heatmap(density, frame):
    density = density.squeeze().cpu().numpy()
    density = cv2.resize(density, (frame.shape[1], frame.shape[0]))
    density = density / (density.max() + 1e-6)
    density = np.uint8(255 * density)
    heat = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heat, 0.4, 0)

# =========================
# EMAIL ALERT (YAGMAIL)
# =========================
import yagmail

def send_alert(count):
    sender_email = "your_email@gmail.com"
    app_password = "YOUR_16_CHAR_APP_PASSWORD"
    receiver_email = "receiver_email@gmail.com"

    yag = yagmail.SMTP(
        user=sender_email,
        password=app_password
    )

    subject = "Crowd Alert"
    contents = f"Alert! Crowd count exceeded threshold.\nCurrent count: {count}"

    yag.send(to=receiver_email, subject=subject, contents=contents)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Crowd Monitoring", layout="wide")
st.title("CSRNet Crowd Monitoring System (Localhost)")

model_A, model_B, device = load_models()

tab1, tab2 = st.tabs(["📷 Webcam Monitoring", "🎥 Video Upload Monitoring"])

# =========================
# TAB 1: WEBCAM
# =========================
with tab1:
    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False
        st.session_state.alert_sent = False

    col1, col2 = st.columns(2)
    if col1.button("Start Webcam"):
        st.session_state.run_cam = True
    if col2.button("Stop Webcam"):
        st.session_state.run_cam = False
        st.session_state.alert_sent = False

    frame_box = st.empty()
    metric_box = st.empty()
    model_box = st.empty()

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)

        while st.session_state.run_cam:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame, device)
            density, count, model_used = infer(tensor, model_A, model_B)
            output = generate_heatmap(density, frame)

            metric_box.metric("Crowd Count", int(count))
            model_box.write(f"Model Used: {model_used}")

            if count > ALERT_THRESHOLD and not st.session_state.alert_sent:
                send_alert(count)
                st.session_state.alert_sent = True
                st.error("ALERT EMAIL SENT")

            frame_box.image(output, channels="BGR")

        cap.release()

# =========================
# TAB 2: VIDEO UPLOAD
# =========================
with tab2:
    uploaded = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_box = st.empty()
        st.session_state.alert_sent = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame, device)
            density, count, model_used = infer(tensor, model_A, model_B)
            output = generate_heatmap(density, frame)

            st.metric("Crowd Count", int(count))
            st.write(f"Model Used: {model_used}")

            if count > ALERT_THRESHOLD and not st.session_state.alert_sent:
                send_alert(count)
                st.session_state.alert_sent = True
                st.error("ALERT EMAIL SENT")

            frame_box.image(output, channels="BGR")

        cap.release()
