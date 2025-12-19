import torch
import torch.nn as nn
import cv2
import numpy as np
import gradio as gr
from torchvision import models, transforms

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSRNet (MATCHES TRAINING ARCHITECTURE)
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# LOAD MODEL (model_5.pth)
print("Loading model...")
model = CSRNet().to(device)
model.load_state_dict(torch.load("model_5.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")


# IMAGE TRANSFORM (SAME AS TRAINING)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# INFERENCE FUNCTION
def infer_image(image):
    if image is None:
        return None, "No image", "No alert"

    original = image.copy()

    # Preprocess
    img = transform(image).unsqueeze(0).to(device).float()

    with torch.no_grad():
        density = model(img)
        density = torch.relu(density)

    # Count
    count = int(density.sum().item())

    # Density → Heatmap
    density_map = density.squeeze().cpu().numpy()
    density_map = cv2.resize(
        density_map,
        (image.shape[1], image.shape[0])
    )

    heatmap = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Alert logic
    if count > 100:
        alert = "🚨 ALERT: Crowd exceeds 100!"
        color = (0, 0, 255)
    else:
        alert = "✅ Crowd within safe limit"
        color = (0, 255, 0)

    cv2.putText(
        overlay,
        f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    return overlay, f"Estimated Count: {count}", alert


# GRADIO INTERFACE (IMAGE UPLOAD)
interface = gr.Interface(
    fn=infer_image,
    inputs=gr.Image(type="numpy", label="Upload Crowd Image"),
    outputs=[
        gr.Image(label="Density Heatmap"),
        gr.Textbox(label="Head Count"),
        gr.Textbox(label="Alert")
    ],
    title="CSRNet Crowd Counting",
    description="Upload an image to see density heatmap, crowd count, and alert if count > 100"
)

interface.launch()
