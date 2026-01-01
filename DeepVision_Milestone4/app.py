import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deep Vision Crowd Monitor", layout="centered")

st.title("Deep Vision Crowd Monitor")
st.write("AI-based Crowd Density Estimation and Overcrowding Detection")

# ---------------- MODEL ----------------
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Load model
@st.cache_resource
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load("model50.pth", map_location="cpu"), strict=False)
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((768, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- CONFIG ----------------
SCALE_FACTOR = 0.12      # you can adjust if needed
ALERT_THRESHOLD = 100   # overcrowd limit

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict density
    with torch.no_grad():
        density = model(img_tensor)

    density = density.squeeze().numpy()
    density[density < 0] = 0

    # Crowd count
    raw_count = density.sum()
    count = int(raw_count * SCALE_FACTOR)

    # Heatmap
    heat = density - density.min()
    heat = heat / (heat.max() + 1e-6)
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    heat = cv2.resize(heat, (image.size[0], image.size[1]))
    overlay = cv2.addWeighted(
        cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
        0.6,
        heat,
        0.4,
        0
    )

    st.image(overlay, caption="Density Heatmap Overlay", use_column_width=True)

    # Display count
    st.subheader(f"Estimated Crowd Count: {count}")

    # Alert
    if count > ALERT_THRESHOLD:
        st.error("🚨 OVER-CROWD ALERT!")
    else:
        st.success("✅ Crowd Level Normal")