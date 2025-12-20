#!/usr/bin/env python3
"""
Pre-Recorded Video Processing with Synthetic Crowd Detection
Demonstrates full pipeline with realistic crowd detection
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from pathlib import Path
from collections import deque
from datetime import datetime
import json
import glob

print("\n" + "="*80)
print("PRE-RECORDED VIDEO PROCESSING - DEMO WITH SYNTHETIC CROWD DATA")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'frame_size': (256, 256),
    'display_size': (640, 480),
    'alert_threshold': 50,
    'smooth_window': 5,
    'fps_output': 20,
    'output_dir': 'video_outputs_demo',
    'save_output': True,
}

Path(CONFIG['output_dir']).mkdir(exist_ok=True)
Path(f"{CONFIG['output_dir']}/videos").mkdir(exist_ok=True)
Path(f"{CONFIG['output_dir']}/statistics").mkdir(exist_ok=True)

# ============================================================================
# SYNTHETIC CROWD DETECTION
# ============================================================================

def generate_synthetic_crowd_data(frame, frame_idx, total_frames):
    """Generate synthetic crowd count based on frame content and temporal patterns"""
    
    # Create realistic crowd patterns
    # Peak in the middle of video, lower at edges
    temporal_factor = 1.0 - abs((frame_idx / total_frames) - 0.5) * 0.6
    
    # Add some variation based on horizontal position
    h, w = frame.shape[:2]
    
    # Detect motion/activity in frame (simple edge detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    motion_intensity = np.sum(edges) / (h * w)
    
    # Base crowd count from motion
    base_count = motion_intensity * 100
    
    # Add temporal variation
    crowd_count = base_count * temporal_factor + np.random.normal(0, 2)
    crowd_count = max(0, crowd_count)
    
    # Create synthetic density map
    density_map = np.zeros((h // 4, w // 4))
    
    if crowd_count > 0:
        # Add Gaussian blobs for crowd areas
        num_blobs = int(crowd_count / 20) + 1
        for _ in range(num_blobs):
            cy = np.random.randint(0, h // 4)
            cx = np.random.randint(0, w // 4)
            radius = int(np.random.uniform(5, 20))
            intensity = crowd_count / num_blobs
            
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x*x + y*y <= radius*radius
            
            cy_min, cy_max = max(0, cy-radius), min(h//4, cy+radius+1)
            cx_min, cx_max = max(0, cx-radius), min(w//4, cx+radius+1)
            
            y_min, y_max = radius-(cy-cy_min), radius+(cy_max-cy)
            x_min, x_max = radius-(cx-cx_min), radius+(cx_max-cx)
            
            if cy_min < cy_max and cx_min < cx_max:
                density_map[cy_min:cy_max, cx_min:cx_max] += intensity * mask[y_min:y_max, x_min:x_max]
    
    return crowd_count, density_map

def create_heatmap_synthetic(density_map, frame_shape):
    """Create heatmap from synthetic density"""
    h, w = frame_shape[:2]
    
    # Upscale density map
    density_upscaled = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    if density_upscaled.max() > 0:
        density_norm = density_upscaled / density_upscaled.max()
    else:
        density_norm = density_upscaled
    
    # Create heatmap
    heatmap = (density_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_color, density_norm

def smooth_count(new_count, history_deque):
    """Smooth predictions"""
    history_deque.append(new_count)
    return np.mean(list(history_deque))

def get_alert_level(count, threshold):
    """Determine alert level"""
    if count < threshold * 0.5:
        return 'LOW', (0, 255, 0), 0
    elif count < threshold:
        return 'MEDIUM', (0, 255, 255), 1
    else:
        return 'HIGH', (0, 0, 255), 2

def get_crowd_level_text(count, threshold):
    """Get crowd level description"""
    percentage = (count / threshold * 100) if threshold > 0 else 0
    
    if percentage < 50:
        return f"Low Crowd ({percentage:.0f}%)", (0, 255, 0)
    elif percentage < 100:
        return f"Medium Crowd ({percentage:.0f}%)", (0, 255, 255)
    else:
        return f"HIGH ALERT ({percentage:.0f}%)", (0, 0, 255)

# ============================================================================
# VIDEO PROCESSOR WITH SYNTHETIC DETECTION
# ============================================================================

class VideoProcessorDemo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        self.total_frames = 0
        self.output_path = None
        self.writer = None
        self.statistics = {
            'video': video_path,
            'total_frames': 0,
            'avg_count': 0,
            'max_count': 0,
            'min_count': float('inf'),
            'alerts': 0,
            'processing_time': 0,
            'frames_data': []
        }
        
    def load_video(self):
        """Load video file"""
        print(f"\n[LOADING] Video: {Path(self.video_path).name}")
        
        if not os.path.exists(self.video_path):
            print(f"[ERROR] File not found: {self.video_path}")
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open video")
                return False
            
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            
            print(f"[SUCCESS] Video loaded")
            print(f"  Resolution: {self.frame_width}x{self.frame_height}")
            print(f"  FPS: {self.fps} | Frames: {self.total_frames}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")
            return False
    
    def setup_output_video(self):
        """Setup output writer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(self.video_path).stem
        self.output_path = f"{CONFIG['output_dir']}/videos/{base_name}_detected_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            CONFIG['fps_output'],
            (self.frame_width, self.frame_height)
        )
        print(f"[SUCCESS] Output will be saved to: {Path(self.output_path).name}")
    
    def annotate_frame(self, frame, crowd_count, alert_code, fps_current, frame_idx):
        """Add annotations"""
        frame_annotated = frame.copy()
        
        # Get level info
        alert_level, alert_color, _ = get_alert_level(crowd_count, CONFIG['alert_threshold'])
        crowd_text, crowd_color = get_crowd_level_text(crowd_count, CONFIG['alert_threshold'])
        
        # Draw info box
        cv2.rectangle(frame_annotated, (10, 10), (450, 120), (0, 0, 0), -1)
        
        if alert_code == 2:
            cv2.rectangle(frame_annotated, (10, 120), (450, 170), (0, 0, 180), -1)
        
        # Draw text
        cv2.putText(frame_annotated, f"Crowd Count: {crowd_count:.1f}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame_annotated, f"Level: {crowd_text}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)
        
        if alert_code == 2:
            cv2.putText(frame_annotated, "[ALERT] High Crowd!", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # FPS and progress
        cv2.putText(frame_annotated, f"FPS: {fps_current:.1f}", 
                   (frame_annotated.shape[1]-250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame_annotated, f"Progress: {frame_idx+1}/{self.total_frames}", 
                   (10, frame_annotated.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return frame_annotated
    
    def process_video(self):
        """Process entire video"""
        if not self.load_video():
            return False
        
        self.setup_output_video()
        
        frame_idx = 0
        count_history = deque(maxlen=CONFIG['smooth_window'])
        start_time = time.time()
        frame_times = deque(maxlen=30)
        
        print(f"\n[PROCESSING] Starting crowd detection...")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            frame_start = time.time()
            
            # Generate synthetic crowd data
            crowd_count, density_map = generate_synthetic_crowd_data(frame, frame_idx, self.total_frames)
            
            # Smooth
            smoothed_count = smooth_count(crowd_count, count_history)
            
            # Get alert
            alert_level, alert_color, alert_code = get_alert_level(smoothed_count, CONFIG['alert_threshold'])
            
            # Create heatmap
            heatmap, _ = create_heatmap_synthetic(density_map, frame.shape)
            frame_heatmap = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            fps_current = 1 / (np.mean(list(frame_times)) + 1e-5) if frame_times else 30
            
            # Annotate
            frame_output = self.annotate_frame(frame_heatmap, smoothed_count, alert_code, fps_current, frame_idx)
            
            # Save stats
            self.statistics['frames_data'].append({
                'frame': frame_idx,
                'count': float(smoothed_count),
                'alert_level': alert_level,
            })
            
            self.statistics['max_count'] = max(self.statistics['max_count'], smoothed_count)
            self.statistics['min_count'] = min(self.statistics['min_count'], smoothed_count)
            
            if alert_code == 2:
                self.statistics['alerts'] += 1
            
            # Write frame
            if self.writer is not None:
                self.writer.write(frame_output)
            
            # Progress
            if (frame_idx + 1) % max(1, self.total_frames // 20) == 0:
                progress = ((frame_idx + 1) / self.total_frames * 100) if self.total_frames > 0 else 0
                print(f"  {progress:5.1f}% | Crowd: {smoothed_count:6.1f} | Level: {alert_level}")
            
            frame_idx += 1
        
        total_time = time.time() - start_time
        
        self.statistics['total_frames'] = frame_idx
        self.statistics['processing_time'] = total_time
        self.statistics['avg_count'] = np.mean([f['count'] for f in self.statistics['frames_data']]) if self.statistics['frames_data'] else 0
        
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        
        print(f"\n[SUCCESS] Processing complete!")
        print(f"  Time: {total_time:.2f}s | Speed: {frame_idx/total_time:.1f} fps")
        
        # Save stats
        self.save_statistics()
        
        return True
    
    def save_statistics(self):
        """Save JSON statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(self.video_path).stem
        stats_path = f"{CONFIG['output_dir']}/statistics/{base_name}_stats_{timestamp}.json"
        
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        # Print summary
        print(f"\n[STATISTICS]")
        print(f"  Avg Crowd:    {self.statistics['avg_count']:.1f}")
        print(f"  Max Crowd:    {self.statistics['max_count']:.1f}")
        print(f"  Min Crowd:    {self.statistics['min_count']:.1f}")
        print(f"  Alerts:       {self.statistics['alerts']}")
        print(f"  Alert Rate:   {self.statistics['alerts']/max(1, self.statistics['total_frames'])*100:.1f}%")
        print(f"\n  Stats file: {Path(stats_path).name}")
        print(f"  Video file: {Path(self.output_path).name if self.output_path else 'N/A'}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("[DEMO MODE - Using Synthetic Crowd Detection]")
    print("(This demonstrates the full pipeline without a trained model)")
    print()
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        # Find first available video
        videos = glob.glob('e:\\DeepVision\\*.mp4')
        videos = [v for v in videos if not any(x in v.lower() for x in ['youtube', 'output', 'detected'])]
        
        if not videos:
            print("[ERROR] No video files found!")
            sys.exit(1)
        
        video_file = videos[0]
    
    print(f"Processing: {Path(video_file).name}\n")
    
    processor = VideoProcessorDemo(video_file)
    processor.process_video()
    
    print("\n[COMPLETE] Demo finished!")
    print(f"Check outputs in: {CONFIG['output_dir']}/\n")
