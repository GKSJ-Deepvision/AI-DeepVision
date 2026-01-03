import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import tempfile

device = torch.device("cpu")

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output_layer(self.backend(self.frontend(x)))

@st.cache_resource
def load_model():
    model = CSRNet().to(device)
    model.load_state_dict(torch.load("model_5.pth", map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("CSRNet Crowd Counting System")

mode = st.radio("Choose input type:", ["Image", "Video"])

if mode == "Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0)

        with torch.no_grad():
            density = torch.relu(model(tensor))

        count = int(density.sum().item())
        dm = cv2.resize(density.squeeze().numpy(), (w, h))
        heatmap = cv2.applyColorMap(
            cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        st.image(overlay, channels="BGR")
        st.success(f"Estimated Crowd Count: {count}")

if mode == "Video":
    file = st.file_uploader("Upload short video", type=["mp4", "avi"])
    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        cap = cv2.VideoCapture(temp.name)

        st.info("Processing video...")
        frames = 0

        while cap.isOpened() and frames < 15:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0)

            with torch.no_grad():
                density = torch.relu(model(tensor))

            count = int(density.sum().item())
            dm = cv2.resize(density.squeeze().numpy(), (w, h))
            heatmap = cv2.applyColorMap(
                cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            st.image(overlay, channels="BGR", caption=f"Count: {count}")
            frames += 1

        cap.release()
