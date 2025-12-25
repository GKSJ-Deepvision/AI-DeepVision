import cv2
from inference import csrnet_infer
from yolo_counter import yolo_people_count
import numpy as np

def smart_infer(frame, use_yolo=False):
    """
    Returns:
    original frame, density map, overlay, count
    """
    original, density_map, overlay, count = csrnet_infer(frame)

    # YOLO fusion for webcam
    if use_yolo:
        yolo_result = yolo_people_count(frame)
        if isinstance(yolo_result, tuple):
            yolo_count, scale = yolo_result
        else:
            yolo_count = yolo_result
        count = max(count, yolo_count)

    return original, density_map, overlay, count
