"""
Preprocessing for ShanghaiTech Part B - Crowd Counting Dataset
Includes:
- Load images from train_data and test_data
- Generate density maps from ground truth annotations
- Apply ImageNet normalization
- Data augmentation for training set
- Save as pickle and NPZ for training
"""

import os
import cv2
import numpy as np
import pickle
from scipy import io as sio
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_density_map_gaussian(img_shape, keypoints, sigma=15):
    """
    Generate density map from keypoints using Gaussian filter
    
    Args:
        img_shape: (height, width) of the original image
        keypoints: list of (y, x) coordinates of people
        sigma: standard deviation for Gaussian filter
    
    Returns:
        density_map: (H, W) density map
    """
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)
    
    for ky, kx in keypoints:
        if 0 <= ky < h and 0 <= kx < w:
            density[int(ky), int(kx)] += 1
    
    # Apply Gaussian filter for smooth density
    density_map = gaussian_filter(density, sigma=sigma)
    return density_map

def normalize_imagenet(image):
    """Apply ImageNet normalization"""
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

def augment_image_and_density(image, density_map, augmentation_params=None):
    """
    Apply same augmentation to both image and density map
    
    Args:
        image: (H, W, 3) image array
        density_map: (H, W) density map
        augmentation_params: dict with augmentation settings
    
    Returns:
        augmented_image, augmented_density
    """
    if augmentation_params is None:
        augmentation_params = {}
    
    augmented_imgs = [image]
    augmented_dens = [density_map]
    
    # Horizontal flip
    if augmentation_params.get('flip_h', True):
        augmented_imgs.append(cv2.flip(image, 1))
        augmented_dens.append(cv2.flip(density_map, 1))
    
    # Vertical flip
    if augmentation_params.get('flip_v', True):
        augmented_imgs.append(cv2.flip(image, 0))
        augmented_dens.append(cv2.flip(density_map, 0))
    
    # Rotation
    if augmentation_params.get('rotate', True):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle = 15
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_img = cv2.warpAffine(image, mat, (w, h))
        rot_dens = cv2.warpAffine(density_map, mat, (w, h))
        augmented_imgs.append(rot_img)
        augmented_dens.append(rot_dens)
    
    return augmented_imgs, augmented_dens

def load_part_b_data():
    """Load ShanghaiTech Part B dataset"""
    root_dir = r"E:\DeepVision\ShanghaiTech\part_B"
    
    train_img_dir = os.path.join(root_dir, "train_data", "images")
    train_gt_dir = os.path.join(root_dir, "train_data", "ground-truth")
    test_img_dir = os.path.join(root_dir, "test_data", "images")
    test_gt_dir = os.path.join(root_dir, "test_data", "ground-truth")
    
    print("=" * 70)
    print(" PREPROCESSING: ShanghaiTech PART B")
    print("=" * 70)
    
    # Load training data
    print("\n [PHASE 1] LOADING TRAINING DATA")
    train_images = []
    train_density_maps = []
    train_counts = []
    
    train_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
    
    for img_file in tqdm(train_files, desc="Processing training images"):
        img_path = os.path.join(train_img_dir, img_file)
        # Convert IMG_1.jpg to GT_IMG_1.mat
        img_num = img_file.replace('IMG_', '').replace('.jpg', '')
        gt_file = f'GT_IMG_{img_num}.mat'
        gt_path = os.path.join(train_gt_dir, gt_file)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Failed to load {img_file}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 256x256
        image_resized = cv2.resize(image, (256, 256))
        
        # Load ground truth
        mat_data = sio.loadmat(gt_path)
        keypoints = mat_data['image_info'][0, 0][0, 0][0]  # Get (y, x) coordinates
        
        # Generate density map at original resolution
        density_orig = generate_density_map_gaussian((orig_h, orig_w), keypoints, sigma=15)
        
        # Resize density map to 64x64
        density_resized = cv2.resize(density_orig, (64, 64))
        
        # Apply scaling factor to preserve density integral
        scale_factor = (orig_w * orig_h) / (64 * 64)
        density_resized = density_resized * scale_factor
        
        # Normalize image (ImageNet)
        image_normalized = normalize_imagenet(image_resized)
        
        train_images.append(image_normalized)
        train_density_maps.append(density_resized)
        train_counts.append(len(keypoints))
    
    print(f" [OK] Training data loaded: {len(train_images)} images")
    
    # Load test data
    print("\n [PHASE 2] LOADING TEST DATA")
    test_images = []
    test_density_maps = []
    test_counts = []
    
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
    
    for img_file in tqdm(test_files, desc="Processing test images"):
        img_path = os.path.join(test_img_dir, img_file)
        # Convert IMG_1.jpg to GT_IMG_1.mat
        img_num = img_file.replace('IMG_', '').replace('.jpg', '')
        gt_file = f'GT_IMG_{img_num}.mat'
        gt_path = os.path.join(test_gt_dir, gt_file)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Failed to load {img_file}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 256x256
        image_resized = cv2.resize(image, (256, 256))
        
        # Load ground truth
        mat_data = sio.loadmat(gt_path)
        keypoints = mat_data['image_info'][0, 0][0, 0][0]  # Get (y, x) coordinates
        
        # Generate density map at original resolution
        density_orig = generate_density_map_gaussian((orig_h, orig_w), keypoints, sigma=15)
        
        # Resize density map to 64x64
        density_resized = cv2.resize(density_orig, (64, 64))
        
        # Apply scaling factor
        scale_factor = (orig_w * orig_h) / (64 * 64)
        density_resized = density_resized * scale_factor
        
        # Normalize image
        image_normalized = normalize_imagenet(image_resized)
        
        test_images.append(image_normalized)
        test_density_maps.append(density_resized)
        test_counts.append(len(keypoints))
    
    print(f" [OK] Test data loaded: {len(test_images)} images")
    
    # Convert to numpy arrays
    X_train = np.array(train_images, dtype=np.float32)
    y_density_train = np.array(train_density_maps, dtype=np.float32)
    y_count_train = np.array(train_counts, dtype=np.float32)
    
    X_test = np.array(test_images, dtype=np.float32)
    y_density_test = np.array(test_density_maps, dtype=np.float32)
    y_count_test = np.array(test_counts, dtype=np.float32)
    
    print(f"\n [PHASE 3] DATA AUGMENTATION (Training set only)")
    
    # Apply augmentation to training data
    X_train_augmented = []
    y_density_train_augmented = []
    y_count_train_augmented = []
    
    for i, (img, dens, count) in enumerate(tqdm(zip(X_train, y_density_train, y_count_train), 
                                                   total=len(X_train), desc="Augmenting")):
        # Add original
        X_train_augmented.append(img)
        y_density_train_augmented.append(dens)
        y_count_train_augmented.append(count)
        
        # Apply augmentations
        aug_imgs, aug_dens = augment_image_and_density(
            (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) * 255).astype(np.uint8),
            dens
        )
        
        for aug_img, aug_dens in zip(aug_imgs[1:], aug_dens[1:]):  # Skip original
            # Normalize augmented image
            aug_img_norm = normalize_imagenet(aug_img / 255.0)
            X_train_augmented.append(aug_img_norm)
            y_density_train_augmented.append(aug_dens)
            y_count_train_augmented.append(count)
    
    X_train_augmented = np.array(X_train_augmented, dtype=np.float32)
    y_density_train_augmented = np.array(y_density_train_augmented, dtype=np.float32)
    y_count_train_augmented = np.array(y_count_train_augmented, dtype=np.float32)
    
    print(f" [OK] Augmentation complete: {len(X_train)} → {len(X_train_augmented)} samples (factor: {len(X_train_augmented)/len(X_train):.1f}x)")
    
    # Create output directory
    output_dir = r"E:\DeepVision\processed_dataset\part_B"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    print(f"\n [PHASE 4] SAVING DATASET")
    data_dict = {
        'X_train': X_train_augmented,
        'X_test': X_test,
        'y_density_train': y_density_train_augmented,
        'y_density_test': y_density_test,
        'y_count_train': y_count_train_augmented,
        'y_count_test': y_count_test,
        'config': {
            'image_size': (256, 256),
            'density_size': (64, 64),
            'normalization': 'ImageNet',
            'gaussian_sigma': 15,
            'augmentation': '3x (original + 2 augmented)'
        }
    }
    
    with open(os.path.join(output_dir, 'part_B_dataset.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)
    
    # Save as NPZ
    np.savez_compressed(
        os.path.join(output_dir, 'part_B_dataset.npz'),
        X_train=X_train_augmented,
        X_test=X_test,
        y_density_train=y_density_train_augmented,
        y_density_test=y_density_test,
        y_count_train=y_count_train_augmented,
        y_count_test=y_count_test
    )
    
    print(f" [OK] Saved: {os.path.join(output_dir, 'part_B_dataset.pkl')}")
    print(f" [OK] Saved: {os.path.join(output_dir, 'part_B_dataset.npz')}")
    
    # Print statistics
    print(f"\n [STATISTICS] DATASET:")
    print(f"  Training samples: {len(X_train_augmented)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Image shape: {X_train_augmented.shape}")
    print(f"  Density shape: {y_density_train_augmented.shape}")
    print(f"  Crowd count range: [{int(np.min(y_count_train_augmented))}, {int(np.max(y_count_train_augmented))}]")
    print(f"  Crowd count mean: {np.mean(y_count_train_augmented):.1f}")
    
    print("\n [DONE] PART B PREPROCESSING COMPLETE!")
    return data_dict

if __name__ == "__main__":
    data = load_part_b_data()
