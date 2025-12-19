import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from torchvision import models
  
#  CSRNET MODEL (Same architecture as training code)
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


#  PREPROCESS FRAME 
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std

    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame

 
 
#  LOAD MODEL
device = "cpu"
model = CSRNet().to(device)

print("Loading model...")
state = torch.load("model_5.pth", map_location=device)
clean_state = {}

for k, v in state.items():
    clean_k = k.replace("output_layer.0", "output_layer")
    clean_state[clean_k] = v

model.load_state_dict(clean_state)
model.eval()
print("Model loaded successfully.")


#  PROCESS FRAME FOR GRADIO
def infer(frame):
    if frame is None:
        return frame, "No frame"

    img_tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    density = pred.squeeze().cpu().numpy()
    density = np.maximum(density, 0)   # REMOVE NEGATIVES
    count = float(density.sum())

    # Heatmap for visualization
    heatmap = (density / density.max()) * 255 if density.max() > 0 else density
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    blended = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    return blended, f"People Count: {count:.2f}"


#  GRADIO UI
interface = gr.Interface(
    fn=infer,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=[
        gr.Image(label="Heatmap Output"),
        gr.Textbox(label="Count")
    ],
    live=True,
    title="Real-Time Crowd Counting (CSRNet)",
)

interface.launch()
