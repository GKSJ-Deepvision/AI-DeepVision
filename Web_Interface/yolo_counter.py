# yolo_counter.py
import cv2
from ultralytics import YOLO

# Load your local YOLOv8 model
model = YOLO(r"D:\Python Projects\DeepVision\yolov8n.pt")  # <-- replace with your path to yolov8.pt

def yolo_people_count(frame):
    """
    Returns tuple: (count, scale)
    """
    results = model.predict(frame, verbose=False)  # predict on single frame
    count = 0
    for r in results:
        # r.boxes contains the detected objects
        count += sum(r.boxes.cls == 0)  # class 0 = person
    scale = max(1, count)
    return count, scale
