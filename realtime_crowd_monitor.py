#!/usr/bin/env python3
"""
Real-Time Crowd Monitoring Pipeline with CSRNet
Integrates trained model with live webcam feed and Gradio UI
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pickle
import threading
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import gradio as gr
from PIL import Image

print("\n" + "="*80)
print("REAL-TIME CROWD MONITORING PIPELINE")
print("="*80 + "\n")

print("[1/4] Loading configuration...")
time.sleep(0.1)  # Brief pause for readability

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'results/csrnet_direct/model.keras',
    'frame_size': (256, 256),
    'alert_threshold': 800,  # Alert if count > this
    'smooth_window': 5,  # Smooth predictions over N frames
    'fps_target': 10,  # Process N frames per second
}

# ============================================================================
# GLOBAL STATE
# ============================================================================

class GlobalState:
    def __init__(self):
        self.latest_frame = None
        self.latest_count = 0
        self.latest_density = None
        self.alert_active = False
        self.frame_history = deque(maxlen=CONFIG['smooth_window'])
        self.processing = False
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.status = "Initializing..."

state = GlobalState()

# ============================================================================
# MODEL LOADING
# ============================================================================

print("[2/4] Loading TensorFlow model (this may take 20-30 seconds)...")
start_load = time.time()
try:
    model = keras.models.load_model(CONFIG['model_path'])
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.1f}s from: {CONFIG['model_path']}")
    print(f"  Parameters: {model.count_params():,}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print(f"  Make sure {CONFIG['model_path']} exists")
    model = None

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_frame(frame):
    """
    Preprocess frame for model input
    
    Args:
        frame: OpenCV frame (BGR, variable size)
    
    Returns:
        processed: Normalized frame (1, 256, 256, 3)
    """
    # Resize to model input size
    resized = cv2.resize(frame, CONFIG['frame_size'])
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Add batch dimension
    batch = np.expand_dims(normalized, axis=0)
    
    return batch, resized

# ============================================================================
# PREDICTION & DENSITY MAP
# ============================================================================

def predict_count(frame_batch):
    """
    Predict crowd count from frame
    
    Args:
        frame_batch: Preprocessed frame (1, 256, 256, 3)
    
    Returns:
        count: Predicted crowd count
    """
    if model is None:
        return 0
    
    try:
        prediction = model.predict(frame_batch, verbose=0)
        count = max(0, float(prediction[0][0]))
        return count
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0

def create_density_map(count, frame_shape):
    """
    Create visualization heatmap of crowd density
    
    Args:
        count: Predicted crowd count
        frame_shape: Shape of original frame
    
    Returns:
        density_visual: Heatmap image
    """
    # Create Gaussian heatmap centered in frame
    h, w = frame_shape[:2]
    
    # Normalize count to visualization intensity (0-255)
    intensity = min(255, int((count / 1000) * 255))
    
    # Create coordinate grids (compatible with newer NumPy)
    y = np.arange(h)[:, np.newaxis]
    x = np.arange(w)[np.newaxis, :]
    cy, cx = h // 2, w // 2
    
    # Distance from center
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cy**2 + cx**2)
    
    # Normalize distance
    dist_norm = dist / max_dist
    
    # Create heatmap (higher intensity = more people)
    heatmap = (1 - dist_norm) * intensity
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_color

def smooth_count(count):
    """
    Smooth predictions using moving average
    """
    state.frame_history.append(count)
    smoothed = np.mean(list(state.frame_history))
    return smoothed

# ============================================================================
# WEBCAM CAPTURE THREAD
# ============================================================================

class WebcamCapture:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start webcam capture thread"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"✓ Webcam started (camera {self.camera_id})")
        return True
    
    def _capture_loop(self):
        """Continuous capture and processing loop"""
        frame_skip = 0
        skip_count = int(30 / CONFIG['fps_target'])  # Skip frames to target FPS
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("✗ Failed to read frame")
                break
            
            state.latest_frame = frame
            
            # Process frame at target FPS
            frame_skip += 1
            if frame_skip >= skip_count:
                frame_skip = 0
                
                # Preprocess
                frame_batch, resized = preprocess_frame(frame)
                
                # Predict
                count = predict_count(frame_batch)
                
                # Smooth
                smoothed_count = smooth_count(count)
                state.latest_count = smoothed_count
                
                # Generate density map
                state.latest_density = create_density_map(smoothed_count, frame.shape)
                
                # Check alert threshold
                state.alert_active = smoothed_count > CONFIG['alert_threshold']
                
                # Update FPS
                state.frame_count += 1
                elapsed = time.time() - state.start_time
                state.fps = state.frame_count / elapsed if elapsed > 0 else 0
                
                # Update status
                status_text = "ALERT! 🔴" if state.alert_active else "Normal ✓"
                state.status = f"{status_text} | Count: {smoothed_count:.0f} | FPS: {state.fps:.1f}"
    
    def stop(self):
        """Stop webcam capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("✓ Webcam stopped")

# Initialize webcam
webcam = WebcamCapture(camera_id=0)

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def get_live_frame():
    """Get current frame with annotations"""
    if state.latest_frame is None:
        # Return blank frame if no input
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        return blank
    
    frame = state.latest_frame.copy()
    
    # Add count text
    cv2.putText(frame, f"Count: {state.latest_count:.0f}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 0), 3)
    
    # Add threshold line
    cv2.putText(frame, f"Threshold: {CONFIG['alert_threshold']}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 165, 255), 2)
    
    # Add alert status
    if state.alert_active:
        cv2.putText(frame, "⚠️ ALERT!", 
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 0, 255), 3)
    
    # Add FPS
    cv2.putText(frame, f"FPS: {state.fps:.1f}", 
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, 
                (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (200, 200, 200), 2)
    
    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb

def get_density_heatmap():
    """Get density heatmap visualization"""
    if state.latest_density is None:
        # Return blank heatmap
        blank = np.ones((256, 256, 3), dtype=np.uint8) * 100
        return blank
    
    heatmap = state.latest_density.copy()
    
    # Add count text on heatmap
    cv2.putText(heatmap, f"Count: {state.latest_count:.0f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb

def get_statistics():
    """Get real-time statistics"""
    total_time = time.time() - state.start_time
    
    stats = f"""
    📊 REAL-TIME CROWD MONITORING STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    👥 CROWD METRICS:
       • Current Count: {state.latest_count:.0f} people
       • Alert Threshold: {CONFIG['alert_threshold']} people
       • Alert Status: {'🔴 ALERT!' if state.alert_active else '✓ Normal'}
    
    ⏱️ PERFORMANCE:
       • Current FPS: {state.fps:.1f} frames/sec
       • Total Frames: {state.frame_count}
       • Uptime: {total_time:.1f} seconds
    
    🎬 PROCESSING:
       • Input Size: {CONFIG['frame_size'][0]}×{CONFIG['frame_size'][1]}
       • Smoothing: {CONFIG['smooth_window']} frames
       • Target FPS: {CONFIG['fps_target']} fps
    
    ⚙️ MODEL:
       • Status: {'✓ Loaded' if model else '✗ Not loaded'}
       • Parameters: {model.count_params():,}
       • Alert Threshold: {CONFIG['alert_threshold']}
    
    💾 SESSION:
       • Start Time: {datetime.fromtimestamp(state.start_time).strftime('%H:%M:%S')}
       • Current Time: {datetime.now().strftime('%H:%M:%S')}
    """
    
    return stats.strip()

def set_alert_threshold(new_threshold):
    """Update alert threshold"""
    CONFIG['alert_threshold'] = new_threshold
    return f"✓ Alert threshold updated to {new_threshold}"

def get_status():
    """Get current status"""
    return state.status

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Real-Time Crowd Monitor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎥 Real-Time Crowd Monitoring Pipeline
        
        **Live crowd counting using CSRNet model with webcam input**
        
        - 📹 Webcam feed with real-time predictions
        - 🔥 Density heatmap visualization
        - 📊 Live statistics and metrics
        - 🚨 Configurable alert thresholds
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📹 Live Feed")
                live_frame = gr.Image(
                    label="Webcam Stream",
                    type="numpy",
                    interactive=False,
                    scale=1
                )
                
            with gr.Column():
                gr.Markdown("### 🔥 Density Heatmap")
                heatmap = gr.Image(
                    label="Crowd Density",
                    type="numpy",
                    interactive=False,
                    scale=1
                )
        
        with gr.Row():
            status = gr.Textbox(
                label="Status",
                interactive=False,
                scale=3,
                lines=1
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Statistics")
                stats = gr.Textbox(
                    label="Real-Time Statistics",
                    interactive=False,
                    lines=15,
                    scale=1
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Controls")
                
                threshold_input = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=CONFIG['alert_threshold'],
                    step=50,
                    label="Alert Threshold",
                    info="Crowd count to trigger alert"
                )
                
                update_btn = gr.Button(
                    "Update Threshold",
                    variant="primary",
                    scale=1
                )
                
                threshold_msg = gr.Textbox(
                    label="Threshold Update",
                    interactive=False,
                    scale=1
                )
        
        # Set up continuous updates
        demo.load(
            get_live_frame,
            outputs=live_frame,
            every=0.5
        )
        
        demo.load(
            get_density_heatmap,
            outputs=heatmap,
            every=0.5
        )
        
        demo.load(
            get_statistics,
            outputs=stats,
            every=1
        )
        
        demo.load(
            get_status,
            outputs=status,
            every=0.5
        )
        
        # Button interactions
        update_btn.click(
            set_alert_threshold,
            inputs=threshold_input,
            outputs=threshold_msg
        )
    
    return demo

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n[3/4] Initializing webcam...")
    
    # Start webcam
    if not webcam.start():
        print("✗ Cannot start webcam. Check if camera is available.")
        exit(1)
    
    # Wait for first frame
    print("  Waiting for first frame...")
    time.sleep(2)
    
    if state.latest_frame is None:
        print("✗ No frames captured. Camera may not be working.")
        webcam.stop()
        exit(1)
    
    print("✓ Pipeline initialized successfully")
    
    # Create and launch Gradio interface
    print("\n[4/4] Launching Gradio interface...")
    print("  Opening at: http://localhost:7860")
    print("  Press Ctrl+C to stop\n")
    
    interface = create_interface()
    
    try:
        interface.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            show_error=True,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n\n[SHUTTING DOWN]")
        print("Stopping webcam...")
        webcam.stop()
        print("✓ Shutdown complete")
