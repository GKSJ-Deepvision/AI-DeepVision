#!/usr/bin/env python3
"""
Real-Time Crowd Monitoring Pipeline with Video File Input
Supports local MP4/MOV files and YouTube videos
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from collections import deque
from datetime import datetime
import subprocess
import sys

import tensorflow as tf
from tensorflow import keras

import gradio as gr
from PIL import Image

print("\n" + "="*80)
print("VIDEO CROWD MONITORING PIPELINE")
print("="*80 + "\n")

print("[1/5] Loading configuration...")
time.sleep(0.1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'results/csrnet_direct/model.keras',
    'frame_size': (256, 256),
    'alert_threshold': 800,
    'smooth_window': 5,
    'fps_target': 10,
    'output_dir': 'crowd_monitor_output',
    'save_output': True,
}

# Create output directory
Path(CONFIG['output_dir']).mkdir(exist_ok=True)

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
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_loaded = False
        self.paused = False

state = GlobalState()

# ============================================================================
# MODEL LOADING
# ============================================================================

print("[2/5] Loading TensorFlow model (this may take 20-30 seconds)...")
start_load = time.time()
try:
    model = keras.models.load_model(CONFIG['model_path'])
    load_time = time.time() - start_load
    print(f"[SUCCESS] Model loaded in {load_time:.1f}s from: {CONFIG['model_path']}")
    print(f"  Parameters: {model.count_params():,}")
except Exception as e:
    print(f"[WARNING] Error loading model: {e}")
    print(f"  Make sure {CONFIG['model_path']} exists")
    model = None

# ============================================================================
# YOUTUBE VIDEO DOWNLOADING
# ============================================================================

def check_youtube_dl():
    """Check if yt-dlp is installed"""
    try:
        import yt_dlp
        return True
    except ImportError:
        return False

def download_youtube_video(youtube_url, output_path='downloaded_video.mp4'):
    """
    Download video from YouTube using yt-dlp
    
    Args:
        youtube_url: URL of YouTube video
        output_path: Path to save downloaded video
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import yt_dlp
        
        print(f"\n[DOWNLOADING VIDEO]")
        print(f"Downloading from: {youtube_url}")
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        print(f"[SUCCESS] Video downloaded to: {output_path}")
        return True
        
    except ImportError:
        print("[ERROR] yt-dlp not installed. Install with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading video: {e}")
        return False

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_frame(frame):
    """
    Preprocess frame for model input
    """
    resized = cv2.resize(frame, CONFIG['frame_size'])
    normalized = resized.astype('float32') / 255.0
    batch = np.expand_dims(normalized, axis=0)
    return batch, resized

# ============================================================================
# PREDICTION & DENSITY MAP
# ============================================================================

def predict_count(frame_batch):
    """Predict crowd count from frame"""
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
    """Create visualization heatmap of crowd density"""
    h, w = frame_shape[:2]
    intensity = min(255, int((count / 1000) * 255))
    
    y = np.arange(h)[:, np.newaxis]
    x = np.arange(w)[np.newaxis, :]
    cy, cx = h // 2, w // 2
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cy**2 + cx**2)
    dist_norm = dist / max_dist
    
    heatmap = (1 - dist_norm) * intensity
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_color

def smooth_count(count):
    """Smooth predictions using moving average"""
    state.frame_history.append(count)
    smoothed = np.mean(list(state.frame_history))
    return smoothed

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

class VideoProcessor:
    """Process video files with crowd monitoring"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.fps = 30
        self.width = 640
        self.height = 480
        self.total_frames = 0
        self.frame_count = 0
        self.out = None
        self.running = False
        self.thread = None
        
    def load_video(self):
        """Load video file"""
        if not Path(self.video_path).exists():
            print(f"[ERROR] Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[SUCCESS] Video loaded: {self.video_path}")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        
        state.total_frames = self.total_frames
        state.video_loaded = True
        
        return True
    
    def setup_output_video(self):
        """Setup output video writer"""
        if not CONFIG['save_output']:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(CONFIG['output_dir']) / f"crowd_monitor_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        print(f"[SUCCESS] Output video will be saved to: {output_path}")
        return output_path
    
    def process_frame(self, frame):
        """Process single frame"""
        if frame is None:
            return frame
        
        frame_copy = frame.copy()
        
        # Preprocess
        frame_batch, resized = preprocess_frame(frame)
        
        # Predict
        count = predict_count(frame_batch)
        smoothed_count = smooth_count(count)
        state.latest_count = smoothed_count
        
        # Check alert
        state.alert_active = smoothed_count > CONFIG['alert_threshold']
        
        # Annotate frame
        color = (0, 0, 255) if state.alert_active else (0, 255, 0)
        
        # Count
        cv2.putText(frame_copy, f"Count: {smoothed_count:.0f}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Threshold
        cv2.putText(frame_copy, f"Threshold: {CONFIG['alert_threshold']}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Alert
        if state.alert_active:
            cv2.putText(frame_copy, "⚠️ ALERT!",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # Red border
            cv2.rectangle(frame_copy, (5, 5), (self.width-5, self.height-5), (0, 0, 255), 5)
        
        # Progress bar
        progress = int((state.current_frame_idx / state.total_frames) * 100) if state.total_frames > 0 else 0
        cv2.putText(frame_copy, f"Progress: {progress}%",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Frame number
        cv2.putText(frame_copy, f"Frame: {state.current_frame_idx}/{state.total_frames}",
                    (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_copy, timestamp,
                    (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        return frame_copy
    
    def process_video(self):
        """Process entire video"""
        if not self.load_video():
            return
        
        output_path = self.setup_output_video()
        
        self.running = True
        self.frame_count = 0
        process_start = time.time()
        
        print("\n[PROCESSING VIDEO]")
        print("Processing frames...")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            state.latest_frame = frame
            state.current_frame_idx = self.frame_count
            
            # Process frame
            annotated = self.process_frame(frame)
            state.latest_frame = annotated
            
            # Save to output video
            if self.out:
                self.out.write(annotated)
            
            self.frame_count += 1
            
            # Update FPS
            elapsed = time.time() - process_start
            state.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Update status
            progress = int((self.frame_count / self.total_frames) * 100)
            state.status = f"Processing: {progress}% | Count: {state.latest_count:.0f} | FPS: {state.fps:.1f}"
            
            # Print progress every 30 frames
            if self.frame_count % 30 == 0:
                print(f"  {progress}% - {self.frame_count}/{self.total_frames} frames processed")
        
        # Cleanup
        self.cap.release()
        if self.out:
            self.out.release()
        
        self.running = False
        
        print(f"\n[SUCCESS] Video processing complete!")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Processing time: {elapsed:.1f}s")
        print(f"  Average FPS: {state.fps:.1f}")
        
        if output_path:
            print(f"  Output saved to: {output_path}")

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def get_live_frame():
    """Get current frame with annotations"""
    if state.latest_frame is None:
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        return blank
    
    frame = state.latest_frame.copy()
    
    # Convert BGR to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    return frame_rgb

def get_density_heatmap():
    """Get density heatmap visualization"""
    if state.latest_density is None:
        blank = np.ones((256, 256, 3), dtype=np.uint8) * 100
        return blank
    
    heatmap = state.latest_density.copy()
    cv2.putText(heatmap, f"Count: {state.latest_count:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap_rgb

def get_statistics():
    """Get real-time statistics"""
    progress = int((state.current_frame_idx / state.total_frames) * 100) if state.total_frames > 0 else 0
    
    # Get model parameters safely
    model_params = f"{model.count_params():,}" if model else "N/A (model not loaded)"
    
    stats = f"""
    📊 VIDEO CROWD MONITORING STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    👥 CROWD METRICS:
       • Current Count: {state.latest_count:.0f} people
       • Alert Threshold: {CONFIG['alert_threshold']} people
       • Alert Status: {'🔴 ALERT!' if state.alert_active else '✓ Normal'}
    
    ⏱️ PROCESSING:
       • Frames Processed: {state.current_frame_idx}/{state.total_frames}
       • Progress: {progress}%
       • Current FPS: {state.fps:.1f} frames/sec
    
    🎬 VIDEO INFO:
       • Status: {'🎬 Playing' if state.video_loaded else '⏸️ Ready'}
       • Frame: {state.current_frame_idx}
       • Total Duration: {state.total_frames / 30:.1f}s (approx)
    
    ⚙️ MODEL:
       • Status: {'✓ Loaded' if model else '✗ Not loaded'}
       • Parameters: {model_params}
       • Alert Threshold: {CONFIG['alert_threshold']}
    
    💾 OUTPUT:
       • Save Output: {'✓ Enabled' if CONFIG['save_output'] else '✗ Disabled'}
       • Output Directory: {CONFIG['output_dir']}
    """
    
    return stats.strip()

def set_alert_threshold(new_threshold):
    """Update alert threshold"""
    CONFIG['alert_threshold'] = new_threshold
    return f"✓ Alert threshold updated to {new_threshold}"

def get_status():
    """Get current status"""
    return state.status

def process_uploaded_video(video_file):
    """Process uploaded video file"""
    if video_file is None:
        return "❌ No video file selected"
    
    try:
        # Get file path
        if isinstance(video_file, dict) and 'name' in video_file:
            video_path = video_file['name']
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        else:
            video_path = str(video_file)
        
        print(f"\n[VIDEO UPLOAD] Processing: {video_path}")
        
        # Create processor and process video
        processor = VideoProcessor(video_path)
        processor.process_video()
        
        return f"✓ Video processed successfully!\nOutput saved to: {CONFIG['output_dir']}"
    
    except Exception as e:
        return f"❌ Error processing video: {str(e)}"

def process_youtube_url(youtube_url):
    """Download and process YouTube video"""
    if not youtube_url or youtube_url.strip() == "":
        return "❌ Please enter a YouTube URL"
    
    try:
        # Check if yt-dlp is installed
        if not check_youtube_dl():
            install_msg = "yt-dlp not found. Install with: pip install yt-dlp"
            print(install_msg)
            return f"❌ {install_msg}"
        
        # Download video
        video_path = "downloaded_youtube_video.mp4"
        if not download_youtube_video(youtube_url, video_path):
            return "❌ Failed to download YouTube video"
        
        # Process video
        processor = VideoProcessor(video_path)
        processor.process_video()
        
        return f"✓ YouTube video processed successfully!\nOutput saved to: {CONFIG['output_dir']}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Video Crowd Monitor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎥 Video Crowd Monitoring Pipeline
        
        **Real-time crowd counting from video files using CSRNet model**
        
        ### Input Options:
        - 📹 Upload local MP4/MOV files
        - 🌐 Download from YouTube (requires yt-dlp)
        - 💾 Process and save annotated output video
        
        ### Features:
        - 👥 Real-time crowd counting
        - 🔥 Density heatmap visualization
        - 🚨 Threshold-based alerts
        - 📊 Live statistics and progress tracking
        - 💾 Automatic output video saving
        """)
        
        with gr.Tabs():
            
            # ==================== LOCAL VIDEO TAB ====================
            with gr.Tab("📁 Local Video"):
                gr.Markdown("### Upload and Process Video File")
                
                video_input = gr.File(
                    label="Upload Video (MP4/MOV)",
                    file_types=[".mp4", ".mov", ".avi"]
                )
                
                process_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    scale=1
                )
                
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                
                process_btn.click(
                    process_uploaded_video,
                    inputs=video_input,
                    outputs=upload_status
                )
            
            # ==================== YOUTUBE TAB ====================
            with gr.Tab("🌐 YouTube"):
                gr.Markdown("### Download from YouTube and Process")
                
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
                
                yt_process_btn = gr.Button(
                    "Download & Process",
                    variant="primary",
                    scale=1
                )
                
                yt_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
                
                gr.Markdown("""
                **Required:** Install yt-dlp first
                ```bash
                pip install yt-dlp
                ```
                """)
                
                yt_process_btn.click(
                    process_youtube_url,
                    inputs=youtube_url,
                    outputs=yt_status
                )
            
            # ==================== PREVIEW TAB ====================
            with gr.Tab("👁️ Live Preview"):
                gr.Markdown("### Live Preview of Current Processing")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📹 Video Frame")
                        live_frame = gr.Image(
                            label="Current Frame",
                            interactive=False
                        )
                        
                    with gr.Column():
                        gr.Markdown("### 🔥 Density Heatmap")
                        heatmap = gr.Image(
                            label="Crowd Density",
                            interactive=False
                        )
                
                with gr.Row():
                    status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        scale=3,
                        lines=1
                    )
                
                # Setup auto-refresh
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
                    get_status,
                    outputs=status,
                    every=0.5
                )
            
            # ==================== STATS TAB ====================
            with gr.Tab("📊 Statistics"):
                gr.Markdown("### Real-Time Statistics")
                
                stats = gr.Textbox(
                    label="Monitoring Statistics",
                    interactive=False,
                    lines=20
                )
                
                demo.load(
                    get_statistics,
                    outputs=stats,
                    every=1
                )
            
            # ==================== SETTINGS TAB ====================
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Configuration")
                
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
                    variant="primary"
                )
                
                threshold_msg = gr.Textbox(
                    label="Update Status",
                    interactive=False
                )
                
                gr.Markdown("### Configuration Info")
                gr.Textbox(
                    value=f"""
Model: {CONFIG['model_path']}
Frame Size: {CONFIG['frame_size'][0]}×{CONFIG['frame_size'][1]}
Smoothing: {CONFIG['smooth_window']} frames
Output Dir: {CONFIG['output_dir']}
Save Output: {CONFIG['save_output']}
                    """,
                    interactive=False,
                    label="Current Settings",
                    lines=6
                )
                
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
    print("\n[3/5] Checking dependencies...")
    
    # Check for yt-dlp
    if check_youtube_dl():
        print("[SUCCESS] yt-dlp installed (YouTube support enabled)")
    else:
        print("[INFO] yt-dlp not found (YouTube support disabled)")
        print("   Install with: pip install yt-dlp")
    
    print("\n[4/5] Initializing interface...")
    time.sleep(1)
    
    print("\n[5/5] Launching Gradio interface...")
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
        print("[SUCCESS] Shutdown complete")
