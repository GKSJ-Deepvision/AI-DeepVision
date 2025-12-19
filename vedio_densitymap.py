import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import tempfile
import shutil
from torchvision import models, transforms

# DEVICE (CPU ONLY)
device = torch.device("cpu")

# CSRNet (MATCHES model_5.pth)
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

# LOAD MODEL
print("Loading model")
model = CSRNet().to(device)
model.load_state_dict(torch.load("model_5.pth", map_location=device))
model.eval()
print("Model loaded")

# TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# VIDEO PROCESSOR (CPU SAFE)
def process_video(video_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    shutil.copy(video_file, tmp.name)

    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * 2)  # 1 frame every 2 seconds

    frame_count = 0
    processed = 0
    MAX_FRAMES = 30

    while cap.isOpened() and processed < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % step != 0:
            continue

        processed += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            density = torch.relu(model(img))

        count = int(density.sum().item())

        density_map = density.squeeze().cpu().numpy()
        density_map = cv2.resize(density_map, (w, h))

        heatmap = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        if count > 100:
            text = f"🚨 ALERT! Count: {count}"
            color = (0, 0, 255)
        else:
            text = f"Count: {count}"
            color = (0, 255, 0)

        cv2.putText(
            overlay, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3
        )

        yield overlay, text

    cap.release()

# GRADIO UI
interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Image(label="Sampled Density Heatmap"),
        gr.Textbox(label="Estimated Crowd Count")
    ],
    title="CSRNet Crowd Counting (CPU-Safe Mode)",
    description="Processes 1 frame every 2 seconds • Max 30 frames • No system freeze"
)

interface.launch()
