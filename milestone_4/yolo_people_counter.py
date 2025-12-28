import torch
from ultralytics import YOLO

# Load YOLOv8 model (small & fast)
yolo_model = YOLO("yolov8n.pt")

def count_people_yolo(frame):
    """
    Counts people using YOLO (class = person)
    Used only for sparse crowd fallback
    """

    results = yolo_model.predict(frame, verbose=False)
    count = 0

    for r in results:
        if r.boxes is not None:
            # class 0 = person
            count += (r.boxes.cls == 0).sum().item()

    return count
