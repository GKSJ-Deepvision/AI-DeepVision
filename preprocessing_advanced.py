"""
Advanced Preprocessing Pipeline for ShanghaiTech Crowd Counting Dataset
- Uses Gaussian filter (scipy.ndimage) for density map generation
- Proper density map resizing with scaling factor
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data augmentation techniques for better generalization
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
import warnings
warnings.filterwarnings('ignore')

# ------------ CONFIGURATION -------------
CONFIG = {
    'dataset_root': r"E:\DeepVision\ShanghaiTech\part_A",
    'output_dir': r"E:\DeepVision\processed_dataset",
    'img_size': (256, 256),  # Target image size
    'density_map_size': (64, 64),  # Downsampled density map
    'gaussian_sigma': 15,  # Gaussian kernel sigma for density maps
    'test_size': 0.2,
    'random_state': 42,
    # ImageNet Normalization
    'imagenet_mean': np.array([0.485, 0.456, 0.406]),
    'imagenet_std': np.array([0.229, 0.224, 0.225]),
    # Augmentation
    'augment_train': True,
    'augmentation_count': 1  # Number of augmented versions per training image
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("=" * 80)
print("🚀 ADVANCED PREPROCESSING PIPELINE FOR CROWD COUNTING")
print("=" * 80)
print("\n✅ Features:")
print("   • Gaussian filter for density map generation (scipy.ndimage)")
print("   • Proper density map resizing with scaling factor")
print("   • Data augmentation for generalization")
print("   • ImageNet normalization")
print("=" * 80)

# ------------ IMAGENET NORMALIZATION FUNCTIONS -----------
def normalize_imagenet(image):
    """
    Apply ImageNet normalization to image.
    
    Args:
        image: numpy array with shape (H, W, 3), values in [0, 1]
    
    Returns:
        normalized_image: ImageNet normalized image
    """
    image_normalized = image.copy().astype(np.float32)
    for c in range(3):
        image_normalized[:, :, c] = (image_normalized[:, :, c] - CONFIG['imagenet_mean'][c]) / CONFIG['imagenet_std'][c]
    return image_normalized

def denormalize_imagenet(image):
    """Reverse ImageNet normalization (for visualization)"""
    image_denorm = image.copy().astype(np.float32)
    for c in range(3):
        image_denorm[:, :, c] = image_denorm[:, :, c] * CONFIG['imagenet_std'][c] + CONFIG['imagenet_mean'][c]
    return np.clip(image_denorm, 0, 1).astype(np.float32)

# ------------ GAUSSIAN DENSITY MAP GENERATION WITH SCIPY -----------
def generate_density_map_gaussian(image_shape, points, sigma=15):
    """
    Generate a Gaussian density map from point annotations using scipy.ndimage.
    
    Args:
        image_shape: (height, width) of the original image
        points: array of (x, y) coordinates of crowd points
        sigma: standard deviation of Gaussian kernel (controls spread)
    
    Returns:
        density_map: 2D array representing crowd density
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # For each point, create a delta image and apply Gaussian filter
    for point in points:
        if len(point) >= 2:
            x, y = int(point[0]), int(point[1])
            
            # Ensure point is within bounds
            if 0 <= x < w and 0 <= y < h:
                # Create a delta image (single point)
                delta = np.zeros((h, w), dtype=np.float32)
                delta[y, x] = 1.0
                
                # Apply Gaussian filter using scipy.ndimage
                gaussian_bump = ndimage.gaussian_filter(delta, sigma=sigma)
                
                # Add to density map
                density_map += gaussian_bump
    
    return density_map

# ------------ DATA AUGMENTATION FUNCTIONS (OpenCV-based) -----------
def augment_image_and_density(image, density_map, aug_type='random'):
    """
    Apply augmentation to both image and density map consistently.
    
    Args:
        image: RGB image (H, W, 3), values in [0, 1]
        density_map: density map (H, W)
        aug_type: type of augmentation ('random', 'flip_h', 'flip_v', 'rotate', 'brightness')
    
    Returns:
        augmented_image, augmented_density
    """
    if aug_type == 'random':
        # Random choice of augmentation
        aug_choice = np.random.choice(['flip_h', 'flip_v', 'rotate', 'brightness', 'blur', 'noise'])
        return augment_image_and_density(image, density_map, aug_choice)
    
    h, w = image.shape[:2]
    aug_image = image.copy()
    aug_density = density_map.copy()
    
    if aug_type == 'flip_h':
        # Horizontal flip
        aug_image = cv2.flip(aug_image, 1)
        aug_density = cv2.flip(aug_density, 1)
    
    elif aug_type == 'flip_v':
        # Vertical flip (less common for crowd scenes)
        if np.random.rand() < 0.3:
            aug_image = cv2.flip(aug_image, 0)
            aug_density = cv2.flip(aug_density, 0)
    
    elif aug_type == 'rotate':
        # Small rotation
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        aug_density = cv2.warpAffine(aug_density, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'brightness':
        # Random brightness adjustment
        hsv = cv2.cvtColor((aug_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        brightness_factor = np.random.uniform(0.8, 1.2)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        aug_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    
    elif aug_type == 'blur':
        # Slight blur
        if np.random.rand() < 0.3:
            aug_image_uint8 = (aug_image * 255).astype(np.uint8)
            aug_image_uint8 = cv2.GaussianBlur(aug_image_uint8, (3, 3), 0)
            aug_image = aug_image_uint8.astype(np.float32) / 255.0
    
    elif aug_type == 'noise':
        # Gaussian noise
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 0.01, aug_image.shape)
            aug_image = np.clip(aug_image + noise, 0, 1)
    
    return aug_image.astype(np.float32), aug_density.astype(np.float32)

# ------------ DATA LOADING FUNCTION WITH PROPER RESIZING -----------
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
                print(f"⚠️ Cannot read image: {img_path}")
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
                    print(f"⚠️ Error reading GT {gt_name}: {e}")
                    points = []
            
            # ========== IMPROVED DENSITY MAP GENERATION ==========
            # 1. Generate density map at ORIGINAL resolution using Gaussian filter
            density_map_original = generate_density_map_gaussian(
                (original_height, original_width),
                points,
                sigma=CONFIG['gaussian_sigma']
            )
            
            # 2. Resize image to target size
            img_resized = cv2.resize(img_rgb, CONFIG['img_size'], interpolation=cv2.INTER_LINEAR)
            
            # 3. Resize density map AND apply scaling factor
            # CRITICAL: When downsampling density, scale to preserve integral
            # Original area = original_width * original_height
            # New area = density_map_size[0] * density_map_size[1]
            scale_factor = (original_width * original_height) / (CONFIG['density_map_size'][1] * CONFIG['density_map_size'][0])
            
            density_resized = cv2.resize(
                density_map_original,
                (CONFIG['density_map_size'][1], CONFIG['density_map_size'][0]),
                interpolation=cv2.INTER_LINEAR
            )
            # Apply scaling factor to preserve density integral
            density_resized = density_resized * scale_factor
            
            # 4. Normalize image to [0, 1]
            img_resized = img_resized.astype(np.float32) / 255.0
            
            # 5. Apply ImageNet normalization
            img_normalized = normalize_imagenet(img_resized)
            
            # Density map stays as is
            density_resized = density_resized.astype(np.float32)
            
            # Store sample
            data_samples.append({
                'image': img_resized,  # Raw normalized image [0, 1]
                'image_imagenet': img_normalized,  # ImageNet normalized
                'density': density_resized,
                'count': len(points),
                'filename': img_name,
                'original_shape': (original_height, original_width),
                'points': points
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return data_samples

# ------------ MAIN PROCESSING PIPELINE =============
print("\n" + "=" * 80)
print("PHASE 1: LOADING DATASET")
print("=" * 80)

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

print("\n" + "=" * 80)
print(f"✅ DATASET LOADED: {len(all_data)} samples")
print("=" * 80)
print(f"  Train samples: {len(train_data)}")
print(f"  Test samples: {len(test_data)}")

# Extract statistics
counts = [sample['count'] for sample in all_data]
print(f"\n  📊 Crowd Count Statistics:")
print(f"     • Min: {min(counts):.0f}")
print(f"     • Max: {max(counts):.0f}")
print(f"     • Mean: {np.mean(counts):.2f}")
print(f"     • Std: {np.std(counts):.2f}")
print(f"     • Median: {np.median(counts):.2f}")

# ------------ PHASE 2: PREPARE TRAINING DATA WITH AUGMENTATION -----------
print("\n" + "=" * 80)
print("PHASE 2: PREPARING TRAINING DATA WITH AUGMENTATION")
print("=" * 80)

# Extract arrays (using ImageNet normalized images)
images = np.array([sample['image_imagenet'] for sample in all_data])
densities = np.array([sample['density'] for sample in all_data])
counts = np.array([sample['count'] for sample in all_data])

print(f"\n✅ Array Shapes (ImageNet Normalized):")
print(f"   • Images: {images.shape}")
print(f"   • Density Maps: {densities.shape}")
print(f"   • Counts: {counts.shape}")

# Split dataset BEFORE augmentation
X_train, X_test, y_density_train, y_density_test, y_count_train, y_count_test = train_test_split(
    images, densities, counts,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state']
)

print(f"\n✅ Train-Test Split (80-20):")
print(f"   • Training: {X_train.shape[0]} samples")
print(f"   • Testing: {X_test.shape[0]} samples")

# ========== DATA AUGMENTATION FOR TRAINING SET ==========
if CONFIG['augment_train']:
    print(f"\n📈 Applying Data Augmentation to Training Set...")
    
    # Store original training data
    X_train_aug = [X_train.copy()]
    y_density_train_aug = [y_density_train.copy()]
    y_count_train_aug = [y_count_train.copy()]
    
    # Generate augmented samples for each training image
    for aug_iter in range(CONFIG['augmentation_count']):
        print(f"\n   Augmentation iteration {aug_iter + 1}/{CONFIG['augmentation_count']}")
        
        X_aug_batch = []
        y_aug_batch = []
        
        # Get raw normalized images from original data
        raw_images = np.array([sample['image'] for sample in all_data])
        _, _, raw_train_imgs, _, _, _ = train_test_split(
            raw_images, densities, counts,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state']
        )
        
        for idx in tqdm(range(len(X_train)), desc=f"Augmenting ({aug_iter + 1})"):
            try:
                # Get raw image and density
                raw_img = raw_train_imgs[idx]
                raw_dens = y_density_train[idx]
                
                # Apply augmentation
                aug_img, aug_dens = augment_image_and_density(raw_img, raw_dens, aug_type='random')
                
                # Re-apply ImageNet normalization after augmentation
                aug_img_norm = normalize_imagenet(aug_img)
                
                X_aug_batch.append(aug_img_norm)
                y_aug_batch.append(aug_dens)
            except Exception as e:
                # Use original if augmentation fails
                X_aug_batch.append(X_train[idx])
                y_aug_batch.append(y_density_train[idx])
        
        X_train_aug.append(np.array(X_aug_batch))
        y_density_train_aug.append(np.array(y_aug_batch))
        y_count_train_aug.append(y_count_train.copy())
    
    # Concatenate all augmented data
    X_train_final = np.concatenate(X_train_aug, axis=0)
    y_density_train_final = np.concatenate(y_density_train_aug, axis=0)
    y_count_train_final = np.concatenate(y_count_train_aug, axis=0)
    
    print(f"\n✅ After augmentation:")
    print(f"   • Training samples: {X_train_final.shape[0]} (original: {X_train.shape[0]})")
    print(f"   • Augmentation boost: {X_train_final.shape[0] / X_train.shape[0]:.1f}x")
else:
    X_train_final = X_train
    y_density_train_final = y_density_train
    y_count_train_final = y_count_train

# ------------ PHASE 3: SAVE PROCESSED DATASET -----------
print("\n" + "=" * 80)
print("PHASE 3: SAVING PROCESSED DATASET")
print("=" * 80)

dataset_dict = {
    'X_train': X_train_final,
    'X_test': X_test,
    'y_density_train': y_density_train_final,
    'y_density_test': y_density_test,
    'y_count_train': y_count_train_final,
    'y_count_test': y_count_test,
    'config': CONFIG,
    'all_samples': all_data,
    'normalization': {
        'type': 'ImageNet',
        'mean': CONFIG['imagenet_mean'].tolist(),
        'std': CONFIG['imagenet_std'].tolist()
    }
}

dataset_path = os.path.join(CONFIG['output_dir'], 'processed_dataset.pkl')
with open(dataset_path, 'wb') as f:
    pickle.dump(dataset_dict, f)

print(f"✅ Dataset saved to: {dataset_path}")

# Save individual arrays as NPZ for easy access
npz_path = os.path.join(CONFIG['output_dir'], 'processed_dataset.npz')
np.savez(
    npz_path,
    X_train=X_train_final,
    X_test=X_test,
    y_density_train=y_density_train_final,
    y_density_test=y_density_test,
    y_count_train=y_count_train_final,
    y_count_test=y_count_test
)
print(f"✅ NumPy arrays saved to: {npz_path}")

# Save metadata
metadata = {
    'total_samples': len(all_data),
    'original_train_samples': len(train_data),
    'original_test_samples': len(test_data),
    'augmented_train_samples': X_train_final.shape[0],
    'test_samples': X_test.shape[0],
    'image_shape': CONFIG['img_size'],
    'density_shape': CONFIG['density_map_size'],
    'gaussian_sigma': CONFIG['gaussian_sigma'],
    'augmentation_enabled': CONFIG['augment_train'],
    'augmentation_count': CONFIG['augmentation_count'],
    'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
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

print("\n" + "=" * 80)
print("✅ ADVANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📁 Output Directory: {CONFIG['output_dir']}")
print(f"\n📋 Features Applied:")
print(f"   ✅ Gaussian filter (scipy.ndimage) for density maps")
print(f"   ✅ Proper density map resizing with scaling factor")
print(f"   ✅ ImageNet normalization applied")
print(f"   ✅ Data augmentation: {CONFIG['augmentation_count']}x augmentation")
print(f"\n📊 Dataset Statistics:")
print(f"   • Original samples: {len(all_data)}")
print(f"   • Training samples (after augmentation): {X_train_final.shape[0]}")
print(f"   • Test samples: {X_test.shape[0]}")
print(f"   • Image shape: {CONFIG['img_size']}")
print(f"   • Density shape: {CONFIG['density_map_size']}")
print(f"\n📂 Files created:")
print(f"   1. processed_dataset.pkl - Full dataset with metadata")
print(f"   2. processed_dataset.npz - NumPy arrays")
print(f"   3. metadata.pkl - Dataset statistics")
print("=" * 80)
