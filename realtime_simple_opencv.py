#!/usr/bin/env python3
"""
Simple Real-Time Crowd Monitoring with OpenCV Only
No external dependencies beyond OpenCV and TensorFlow
Useful for testing without Gradio
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

import tensorflow as tf
from tensorflow import keras

print("\n" + "="*80)
print("REAL-TIME CROWD MONITOR (OpenCV Version)")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'results/csrnet_direct/model.keras',
    'frame_size': (256, 256),
    'alert_threshold': 800,
    'smooth_window': 5,
    'display_size': (1280, 720),
}

# ============================================================================
# MODEL LOADING
# ============================================================================

print("[LOADING MODEL]")
try:
    model = keras.models.load_model(CONFIG['model_path'])
    print(f"✓ Model loaded")
    print(f"  Parameters: {model.count_params():,}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# ============================================================================
# FUNCTIONS
# ============================================================================

def preprocess_frame(frame):
    """Preprocess frame for model"""
    resized = cv2.resize(frame, CONFIG['frame_size'])
    normalized = resized.astype('float32') / 255.0
    batch = np.expand_dims(normalized, axis=0)
    return batch, resized

def predict_count(frame_batch):
    """Predict crowd count"""
    if model is None:
        return 0
    try:
        prediction = model.predict(frame_batch, verbose=0)
        return max(0, float(prediction[0][0]))
    except:
        return 0

def create_density_heatmap(count, shape):
    """Create density heatmap"""
    h, w = shape[:2]
    intensity = min(255, int((count / 1000) * 255))
    
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cy**2 + cx**2)
    dist_norm = dist / max_dist
    
    heatmap = (1 - dist_norm) * intensity
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_color

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("[INITIALIZING WEBCAM]")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Webcam initialized")
    print("[STARTING REAL-TIME PROCESSING]")
    print("Press 'q' to quit, 'r' to reset, 't' to change threshold\n")
    
    frame_count = 0
    start_time = time.time()
    count_history = deque(maxlen=CONFIG['smooth_window'])
    alert_threshold = CONFIG['alert_threshold']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Preprocess
        frame_batch, resized = preprocess_frame(frame)
        
        # Predict
        count = predict_count(frame_batch)
        count_history.append(count)
        smoothed_count = np.mean(list(count_history))
        
        # Check alert
        alert = smoothed_count > alert_threshold
        
        # FPS calculation
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        
        # Create visualization
        display = cv2.resize(frame, CONFIG['display_size'])
        
        # Add information
        info_y = 50
        cv2.putText(display, f"Crowd Count: {smoothed_count:.0f}", 
                    (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        cv2.putText(display, f"Threshold: {alert_threshold}", 
                    (30, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        
        if alert:
            cv2.putText(display, "⚠️ ALERT: Crowd Exceeds Threshold!", 
                        (30, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Draw red border for alert
            cv2.rectangle(display, (10, 10), 
                         (CONFIG['display_size'][0]-10, CONFIG['display_size'][1]-10), 
                         (0, 0, 255), 5)
        
        cv2.putText(display, f"FPS: {fps:.1f}", 
                    (30, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(display, f"Time: {datetime.now().strftime('%H:%M:%S')}", 
                    (30, info_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Resize and display heatmap
        heatmap = create_density_heatmap(smoothed_count, frame)
        heatmap_display = cv2.resize(heatmap, (320, 320))
        
        # Combine frame and heatmap
        h = CONFIG['display_size'][1]
        combined = np.zeros((h + 50, CONFIG['display_size'][0], 3), dtype=np.uint8)
        combined[:h, :] = display
        
        # Add heatmap in corner
        heatmap_resized = cv2.resize(heatmap, (150, 150))
        combined[h-160:h-10, CONFIG['display_size'][0]-160:CONFIG['display_size'][0]-10] = heatmap_resized
        
        # Display
        cv2.imshow('Real-Time Crowd Monitor', combined)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nShutting down...")
            break
        elif key == ord('r'):
            print("Reset statistics")
            frame_count = 0
            start_time = time.time()
        elif key == ord('t'):
            print(f"Current threshold: {alert_threshold}")
            print("Enter new threshold (e.g., 500): ", end='')
            try:
                new_threshold = int(input())
                alert_threshold = new_threshold
                print(f"✓ Threshold updated to {alert_threshold}")
            except:
                print("Invalid input")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Shutdown complete")

if __name__ == "__main__":
    main()
