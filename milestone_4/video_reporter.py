import cv2
import os
import time

def generate_report(video_path, max_count):
    """
    Generates report for uploaded video:
    - Duration
    - Upload time
    - Maximum crowd count
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total_frames / fps if fps else 0
    cap.release()

    upload_time = time.ctime(os.path.getctime(video_path))

    return {
        "duration": f"{duration:.2f} seconds",
        "uploaded_time": upload_time,
        "max_crowd": int(max_count)
    }
