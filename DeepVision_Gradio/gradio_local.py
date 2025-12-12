import cv2
import torch
import numpy as np
import gradio as gr
import torchvision.transforms as T

# ---------- Load model ----------
class CSRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.conv(x)

model = CSRNet()
model.load_state_dict(torch.load("model50.pth", map_location="cpu"), strict=False)

model.eval()

# ---------- Preprocessing ----------
transform = T.Compose([
    T.ToTensor()
])

THRESHOLD = 50

def predict(frame):
    img = transform(frame).unsqueeze(0)
    with torch.no_grad():
        density = model(img).squeeze().numpy()
    density = np.maximum(density, 0)
    return density

def heatmap(density):
    d = density - density.min()
    d = (d / (d.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)

# ---------- Gradio function ----------
def process(frame):
    density = predict(frame)
    count = density.sum()

    heat = heatmap(density)
    heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 0.5, heat, 0.5, 0)

    if count > THRESHOLD:
        cv2.putText(overlay, "OVER-CROWD ALERT!", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(overlay, f"Count: {count:.1f}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return overlay, f"{count:.1f}"

# ---------- Interface ----------
iface = gr.Interface(
    fn=process,
   inputs=gr.Image(sources=["webcam"]),

    outputs=[gr.Image(), gr.Textbox(label="Crowd Count")],
    live=True
)

iface.launch()
