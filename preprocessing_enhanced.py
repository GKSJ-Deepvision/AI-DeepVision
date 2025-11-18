"""
Enhanced Preprocessing Pipeline for ShanghaiTech Crowd Counting Dataset
- Loads images and ground truth annotations
- Generates Gaussian density maps
- Performs data augmentation
- Saves processed dataset for model training
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# ------------ CONFIGURATION -------------
CONFIG = {
    'dataset_root': r"E:\DeepVision\ShanghaiTech\part_A",
    'output_dir': r"E:\DeepVision\processed_dataset",
    'img_size': (256, 256),  # Target image size
    'density_map_size': (64, 64),  # Downsampled density map
    'gaussian_sigma': 15,  # Gaussian kernel sigma for density maps
    'test_size': 0.2,
    'random_state': 42
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("=" * 70)
print("🚀 ENHANCED PREPROCESSING PIPELINE FOR CROWD COUNTING")
print("=" * 70)

# ------------ DENSITY MAP GENERATION FUNCTION -------------
def generate_density_map(image_shape, points, sigma=15):
    """
    Generate a Gaussian density map from point annotations.
    
    Args:
        image_shape: (height, width) of the original image
        points: array of (x, y) coordinates of crowd points
        sigma: standard deviation of Gaussian kernel
    
    Returns:
        density_map: 2D array representing crowd density
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # Add Gaussian bump at each point
    for point in points:
        if len(point) >= 2:
            x, y = int(point[0]), int(point[1])
            # Ensure point is within bounds
            if 0 <= x < w and 0 <= y < h:
                # Create Gaussian kernel
                y_min = max(0, y - 3*sigma)
                y_max = min(h, y + 3*sigma + 1)
                x_min = max(0, x - 3*sigma)
                x_max = min(w, x + 3*sigma + 1)
                
                # Generate 2D Gaussian
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                
                # Add to density map
                density_map[y_min:y_max, x_min:x_max] += gaussian
    
    return density_map

# ------------ DATA LOADING FUNCTION -------------
def load_shanghaitech_data_with_density(images_path, gt_path, config):
    """
    Load images, extract crowd points, and generate density maps.
    
    Returns:
        list of dicts: {'image': img, 'density': density_map, 'count': count, 'filename': name}
    """
    data_samples = []
    
    if not os.path.exists(images_path):
        print(f"❌ Images path does not exist: {images_path}")
        return data_samples
    
    img_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])
    
    print(f"\n🔄 Loading {len(img_files)} images from: {os.path.basename(images_path)}\n")
    
    for img_name in tqdm(img_files, desc=f"Processing {os.path.basename(images_path)}"):
        try:
            img_path = os.path.join(images_path, img_name)
            gt_name = img_name.replace('.jpg', '.mat')
            gt_path_file = os.path.join(gt_path, f"GT_{gt_name}")
            
            # Load original image
            img_original = cv2.imread(img_path)
            if img_original is None:
                print(f"⚠ Cannot read image: {img_path}")
                continue
            
            original_height, original_width = img_original.shape[:2]
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            
            # Load ground truth and extract points
            points = []
            if os.path.exists(gt_path_file):
                try:
                    mat_data = sio.loadmat(gt_path_file)
                    
                    # Extract annotation points (varies by dataset version)
                    if 'image_info' in mat_data:
                        annotations = mat_data['image_info'][0][0][0][0][0]
                        points = [(p[0], p[1]) for p in annotations if len(p) >= 2]
                    elif 'annPoints' in mat_data:
                        points_data = mat_data['annPoints']
                        points = [(p[0], p[1]) for p in points_data if len(p) >= 2]
                    else:
                        # Fallback: search for any array with point-like data
                        for key in mat_data.keys():
                            if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                                if mat_data[key].size > 0 and mat_data[key].ndim == 2:
                                    if mat_data[key].shape[1] >= 2:
                                        points = [(p[0], p[1]) for p in mat_data[key]]
                                        break
                except Exception as e:
                    print(f"⚠ Error reading GT {gt_name}: {e}")
                    points = []
            
            # Generate density map for original size
            density_map_original = generate_density_map(
                (original_height, original_width),
                points,
                sigma=CONFIG['gaussian_sigma']
            )
            
            # Resize image and density map
            img_resized = cv2.resize(img_rgb, CONFIG['img_size'], interpolation=cv2.INTER_LINEAR)
            density_resized = cv2.resize(
                density_map_original,
                (CONFIG['density_map_size'][1], CONFIG['density_map_size'][0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize
            img_resized = img_resized.astype(np.float32) / 255.0
            density_resized = density_resized.astype(np.float32)
            
            # Store sample
            data_samples.append({
                'image': img_resized,
                'density': density_resized,
                'count': len(points),
                'filename': img_name,
                'original_shape': (original_height, original_width),
                'points': points
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            continue
    
    return data_samples

# ------------ MAIN PROCESSING PIPELINE -------------
print("\n" + "=" * 70)
print("PHASE 1: LOADING DATASET")
print("=" * 70)

# Load test data
test_images_path = os.path.join(CONFIG['dataset_root'], "test_data", "images")
test_gt_path = os.path.join(CONFIG['dataset_root'], "test_data", "ground-truth")
test_data = load_shanghaitech_data_with_density(test_images_path, test_gt_path, CONFIG)

# Load train data
train_images_path = os.path.join(CONFIG['dataset_root'], "train_data", "images")
train_gt_path = os.path.join(CONFIG['dataset_root'], "train_data", "ground-truth")
train_data = load_shanghaitech_data_with_density(train_images_path, train_gt_path, CONFIG)

# Combine
all_data = train_data + test_data

print("\n" + "=" * 70)
print(f"DATASET SUMMARY: {len(all_data)} samples loaded")
print("=" * 70)
print(f"  Train samples: {len(train_data)}")
print(f"  Test samples: {len(test_data)}")

# Extract statistics
counts = [sample['count'] for sample in all_data]
print(f"\n  Crowd Count Statistics:")
print(f"    - Min: {min(counts):.0f}")
print(f"    - Max: {max(counts):.0f}")
print(f"    - Mean: {np.mean(counts):.2f}")
print(f"    - Std: {np.std(counts):.2f}")
print(f"    - Median: {np.median(counts):.2f}")

# ------------ PHASE 2: PREPARE TRAINING DATA -------------
print("\n" + "=" * 70)
print("PHASE 2: PREPARING TRAINING DATA")
print("=" * 70)

# Extract arrays
images = np.array([sample['image'] for sample in all_data])
densities = np.array([sample['density'] for sample in all_data])
counts = np.array([sample['count'] for sample in all_data])

print(f"\nArray Shapes:")
print(f"  Images: {images.shape} - (samples, H={CONFIG['img_size'][0]}, W={CONFIG['img_size'][1]}, C=3)")
print(f"  Density Maps: {densities.shape} - (samples, H={CONFIG['density_map_size'][0]}, W={CONFIG['density_map_size'][1]})")
print(f"  Counts: {counts.shape}")

# Split dataset
X_train, X_test, y_density_train, y_density_test, y_count_train, y_count_test = train_test_split(
    images, densities, counts,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state']
)

print(f"\nTrain-Test Split (80-20):")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# ------------ PHASE 3: SAVE PROCESSED DATASET -------------
print("\n" + "=" * 70)
print("PHASE 3: SAVING PROCESSED DATASET")
print("=" * 70)

dataset_dict = {
    'X_train': X_train,
    'X_test': X_test,
    'y_density_train': y_density_train,
    'y_density_test': y_density_test,
    'y_count_train': y_count_train,
    'y_count_test': y_count_test,
    'config': CONFIG,
    'all_samples': all_data
}

dataset_path = os.path.join(CONFIG['output_dir'], 'processed_dataset.pkl')
with open(dataset_path, 'wb') as f:
    pickle.dump(dataset_dict, f)

print(f"✅ Dataset saved to: {dataset_path}")

# Save individual arrays as NPZ for easy access
npz_path = os.path.join(CONFIG['output_dir'], 'processed_dataset.npz')
np.savez(
    npz_path,
    X_train=X_train,
    X_test=X_test,
    y_density_train=y_density_train,
    y_density_test=y_density_test,
    y_count_train=y_count_train,
    y_count_test=y_count_test
)
print(f"✅ NumPy arrays saved to: {npz_path}")

# Save metadata
metadata = {
    'total_samples': len(all_data),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'image_shape': CONFIG['img_size'],
    'density_shape': CONFIG['density_map_size'],
    'count_stats': {
        'min': float(np.min(counts)),
        'max': float(np.max(counts)),
        'mean': float(np.mean(counts)),
        'std': float(np.std(counts)),
        'median': float(np.median(counts))
    }
}

metadata_path = os.path.join(CONFIG['output_dir'], 'metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Metadata saved to: {metadata_path}")

print("\n" + "=" * 70)
print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nOutput Directory: {CONFIG['output_dir']}")
print(f"Files created:")
print(f"  1. processed_dataset.pkl - Full dataset with metadata")
print(f"  2. processed_dataset.npz - NumPy arrays")
print(f"  3. metadata.pkl - Dataset statistics")
print("=" * 70)
