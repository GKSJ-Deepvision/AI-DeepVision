import cv2
import torch
import numpy as np
import torchvision.transforms as T

# ---------------- MODEL ----------------
class CSRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

model = CSRNet()
model.load_state_dict(torch.load("model50.pth", map_location="cpu"), strict=False)
model.eval()

transform = T.Compose([T.ToTensor()])

# ---------------- FUNCTIONS ----------------
def predict_density(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        density = model(img)

    return density.squeeze().numpy()

def make_heatmap(density):
    density = density - density.min()
    if density.max() != 0:
        density = density / density.max()
    density = np.uint8(255 * density)
    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return heatmap

# ---------------- VIDEO ----------------
VIDEO_PATH = "crowd_video.mp4"
THRESHOLD = 50

cap = cv2.VideoCapture(VIDEO_PATH)
print(cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    density = predict_density(frame)
    count = int(density.sum())

    heat = make_heatmap(density)
    heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)

    cv2.putText(overlay, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if count > THRESHOLD:
        cv2.putText(overlay, "OVER-CROWD ALERT!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Pre-recorded Crowd Video Analysis", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()