import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request
from torchvision import models, transforms

# ================= BASIC CONFIG =================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= CSRNet MODEL =================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ================= LOAD TRAINED MODEL =================
print("Loading model_5.pth...")
model = CSRNet().to(device)
model.load_state_dict(torch.load("model_5.pth", map_location=device))
model.eval()
print("Model loaded successfully")

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= VIDEO PROCESSING =================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # safety fallback

    frame_step = int(fps * 2)  # process every 2 seconds
    frame_step = max(1, frame_step)

    frame_count = 0
    processed_frames = 0
    total_count = 0
    heatmap_saved = False

    while cap.isOpened() and processed_frames < 10:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_step != 0:
            continue

        processed_frames += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            density = torch.relu(model(tensor))

        count = int(density.sum().item())
        total_count += count

        # Save ONE heatmap image
        if not heatmap_saved:
            density_map = density.squeeze().cpu().numpy()
            density_map = cv2.resize(density_map, (w, h))

            heatmap = cv2.normalize(
                density_map, None, 0, 255, cv2.NORM_MINMAX
            )
            heatmap = cv2.applyColorMap(
                heatmap.astype(np.uint8), cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, "output.jpg"), overlay)

            heatmap_saved = True

    cap.release()

    avg_count = total_count // max(1, processed_frames)
    return avg_count

# ================= FLASK ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
                result = process_video(save_path)
                image_path = "uploads/output.jpg"

    return render_template(
        "index.html",
        result=result,
        image=image_path
    )

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)