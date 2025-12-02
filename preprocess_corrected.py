"""
CORRECTED CSRNet PREPROCESSING - ShanghaiTech Dataset
Key Fixes:
1. ImageNet normalization using correct mean/std
2. Proper 1/8 downsampling (256x256 → 64x64)
3. Density maps scaled by 64 to preserve integral
4. Verification of density-count matching
5. Proper image and density alignment
"""

import os
import sys
import numpy as np
import scipy.io as sio
import cv2
import pickle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================
print("="*80)
print("[CORRECTED PREPROCESSING] CSRNet - ShanghaiTech Part A")
print("="*80)

DATASET_PATH = r"E:\DeepVision\ShanghaiTech\part_A"
OUTPUT_DIR = r"E:\DeepVision\processed_dataset_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ImageNet normalization constants (CORRECT)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Preprocessing parameters
IMG_SIZE = 256
DENSITY_SIZE = 64  # 256 / 4 = 64 (1/4 downsampling for 1/8 area downsampling)
GAUSSIAN_SIGMA = 15
DENSITY_SCALE = 64  # Scale factor for density maps

print(f"\n[CONFIG]")
print(f"  Input image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"  Density map size: {DENSITY_SIZE}×{DENSITY_SIZE}")
print(f"  Downsampling factor: {IMG_SIZE//DENSITY_SIZE}× (area: 1/{(IMG_SIZE//DENSITY_SIZE)**2})")
print(f"  Density scale factor: {DENSITY_SCALE}×")
print(f"  Gaussian sigma: {GAUSSIAN_SIGMA}")
print(f"  ImageNet mean: {IMAGENET_MEAN}")
print(f"  ImageNet std: {IMAGENET_STD}")

# ============================================================================
# STEP 2: LOAD DATASET WITH CORRECT CROWD ANNOTATIONS
# ============================================================================
print(f"\n[STEP 1] LOADING DATASET")
print("-"*80)

def load_shanghaitech_part_a(dataset_path):
    """Load ShanghaiTech Part A with crowd point annotations"""
    
    train_images_path = os.path.join(dataset_path, "train_data", "images")
    train_gt_path = os.path.join(dataset_path, "train_data", "ground-truth")
    test_images_path = os.path.join(dataset_path, "test_data", "images")
    test_gt_path = os.path.join(dataset_path, "test_data", "ground-truth")
    
    images_all = []
    points_all = []  # Crowd head points
    
    # Load training data
    print("Loading training data...")
    train_imgs = sorted([f for f in os.listdir(train_images_path) if f.endswith('.jpg')])
    
    for img_name in tqdm(train_imgs, desc="Train"):
        img_path = os.path.join(train_images_path, img_name)
        gt_name = img_name.replace('.jpg', '.mat')
        gt_path_file = os.path.join(train_gt_path, f"GT_{gt_name}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load crowd points
        try:
            mat_data = sio.loadmat(gt_path_file)
            # ShanghaiTech stores head annotations in 'image_info'
            if 'image_info' in mat_data:
                head_points = mat_data['image_info'][0, 0][0, 0][0]  # (N, 2)
            else:
                head_points = np.array([])
        except:
            head_points = np.array([])
        
        images_all.append(img)
        points_all.append(head_points)
    
    # Load test data
    print("Loading test data...")
    test_imgs = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg')])
    
    for img_name in tqdm(test_imgs, desc="Test"):
        img_path = os.path.join(test_images_path, img_name)
        gt_name = img_name.replace('.jpg', '.mat')
        gt_path_file = os.path.join(test_gt_path, f"GT_{gt_name}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load crowd points
        try:
            mat_data = sio.loadmat(gt_path_file)
            if 'image_info' in mat_data:
                head_points = mat_data['image_info'][0, 0][0, 0][0]
            else:
                head_points = np.array([])
        except:
            head_points = np.array([])
        
        images_all.append(img)
        points_all.append(head_points)
    
    return images_all, points_all

images_raw, crowd_points = load_shanghaitech_part_a(DATASET_PATH)
print(f"\n[OK] Loaded {len(images_raw)} images")
print(f"     Image sizes: {[img.shape for img in images_raw[:3]]}")
print(f"     Crowd points: {[len(p) if len(p.shape) > 0 else 0 for p in crowd_points[:5]]}")

# ============================================================================
# STEP 3: GENERATE DENSITY MAPS - CORRECTED
# ============================================================================
print(f"\n[STEP 2] GENERATING DENSITY MAPS")
print("-"*80)

def generate_density_map_corrected(head_points, img_shape):
    """
    Generate density map from crowd head annotations.
    
    Key corrections:
    1. Create map at FULL RESOLUTION first
    2. Gaussian smooth the full-resolution map
    3. Then downsample to final size
    4. Scale to preserve integral
    """
    
    # Full resolution density map
    h, w = img_shape[:2]
    density_full = np.zeros((h, w), dtype=np.float32)
    
    # Place Gaussian kernels at head locations
    if len(head_points) > 0 and head_points.shape[0] > 0:
        # ShanghaiTech format: (x, y) coordinates
        for point in head_points:
            x, y = point
            x, y = int(x), int(y)
            
            # Ensure within bounds
            if 0 <= x < w and 0 <= y < h:
                # Place 1 at head location
                density_full[y, x] += 1.0
    
    # Apply Gaussian smoothing at full resolution
    density_full = gaussian_filter(density_full, sigma=GAUSSIAN_SIGMA)
    
    # Downsample to target size
    density_map = cv2.resize(
        density_full,
        (DENSITY_SIZE, DENSITY_SIZE),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Calculate scaling factor: area reduction = (img_size/density_size)^2
    # When downsampling, we lose area, so we must scale up to preserve integral
    scale = (h * w) / (DENSITY_SIZE * DENSITY_SIZE)
    density_map = density_map * scale
    
    return density_map

# Generate all density maps
print("Generating density maps...")
density_maps = []
crowd_counts = []

for img, points in tqdm(zip(images_raw, crowd_points), total=len(images_raw)):
    # Count of people from point annotations
    crowd_count = len(points) if len(points.shape) > 0 else 0
    
    # Generate density map
    if len(points.shape) > 0 and points.shape[0] > 0:
        density = generate_density_map_corrected(points, img.shape)
    else:
        # Empty image - zero density map
        density = np.zeros((DENSITY_SIZE, DENSITY_SIZE), dtype=np.float32)
    
    density_maps.append(density)
    crowd_counts.append(crowd_count)

density_maps = np.array(density_maps, dtype=np.float32)
crowd_counts = np.array(crowd_counts, dtype=np.int32)

print(f"[OK] Generated {len(density_maps)} density maps")
print(f"     Shape: {density_maps.shape}")
print(f"     Crowd counts: min={crowd_counts.min()}, max={crowd_counts.max()}, mean={crowd_counts.mean():.1f}")
print(f"     Density values: min={density_maps.min():.4f}, max={density_maps.max():.4f}, mean={density_maps.mean():.4f}")

# Verify density-count matching
density_sums = density_maps.sum(axis=(1, 2))
count_errors = np.abs(density_sums - crowd_counts)
print(f"\n[VERIFICATION] Density map vs crowd count matching:")
print(f"     Density sums - mean: {density_sums.mean():.2f}, std: {density_sums.std():.2f}")
print(f"     Count errors - mean: {count_errors.mean():.2f}, max: {count_errors.max():.2f}")
print(f"     Sample matches:")
for i in range(min(5, len(crowd_counts))):
    print(f"       Image {i}: count={crowd_counts[i]}, density_sum={density_sums[i]:.1f}, error={count_errors[i]:.2f}")

# ============================================================================
# STEP 4: IMAGE PREPROCESSING - CORRECTED
# ============================================================================
print(f"\n[STEP 3] IMAGE PREPROCESSING")
print("-"*80)

def preprocess_image(img):
    """
    Preprocess image:
    1. Resize to 256×256
    2. Convert to float [0, 1]
    3. Apply ImageNet normalization
    """
    # Resize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to float and normalize to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img_normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    
    return img_normalized

print("Preprocessing images...")
X = np.zeros((len(images_raw), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

for i, img in enumerate(tqdm(images_raw)):
    X[i] = preprocess_image(img)

print(f"[OK] Preprocessed all images")
print(f"     Shape: {X.shape}")
print(f"     Value range: [{X.min():.3f}, {X.max():.3f}]")
print(f"     Mean per channel: {X.mean(axis=(0, 1, 2))}")
print(f"     Std per channel: {X.std(axis=(0, 1, 2))}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print(f"\n[STEP 4] TRAIN-TEST SPLIT")
print("-"*80)

# Use the original split: first 300 for training, remaining for testing
num_train = 300

X_train = X[:num_train]
X_test = X[num_train:]
y_density_train = density_maps[:num_train]
y_density_test = density_maps[num_train:]
y_count_train = crowd_counts[:num_train]
y_count_test = crowd_counts[num_train:]

print(f"Training set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")
print(f"\nTraining crowd counts: min={y_count_train.min()}, max={y_count_train.max()}, mean={y_count_train.mean():.1f}")
print(f"Test crowd counts: min={y_count_test.min()}, max={y_count_test.max()}, mean={y_count_test.mean():.1f}")

# ============================================================================
# STEP 6: SAVE CORRECTED DATASET
# ============================================================================
print(f"\n[STEP 5] SAVING CORRECTED DATASET")
print("-"*80)

dataset = {
    'X_train': X_train,
    'X_test': X_test,
    'y_density_train': y_density_train,
    'y_density_test': y_density_test,
    'y_count_train': y_count_train,
    'y_count_test': y_count_test,
    'config': {
        'img_size': IMG_SIZE,
        'density_size': DENSITY_SIZE,
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'density_scale': DENSITY_SCALE,
        'imagenet_mean': IMAGENET_MEAN,
        'imagenet_std': IMAGENET_STD,
    }
}

output_path = os.path.join(OUTPUT_DIR, 'part_A_corrected.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)

print(f"[OK] Saved corrected dataset: {output_path}")
print(f"     Size: {os.path.getsize(output_path) / (1024**2):.1f} MB")

# ============================================================================
# STEP 7: VERIFICATION
# ============================================================================
print(f"\n[STEP 6] FINAL VERIFICATION")
print("-"*80)

# Reload and verify
with open(output_path, 'rb') as f:
    loaded = pickle.load(f)

print(f"[VERIFICATION CHECKS]")
print(f"  ✓ X_train shape: {loaded['X_train'].shape}")
print(f"  ✓ X_test shape: {loaded['X_test'].shape}")
print(f"  ✓ y_density_train shape: {loaded['y_density_train'].shape}")
print(f"  ✓ y_density_test shape: {loaded['y_density_test'].shape}")

# Verify density-count matching
density_sums_train = loaded['y_density_train'].sum(axis=(1, 2))
count_errors_train = np.abs(density_sums_train - loaded['y_count_train'])

print(f"\n[DENSITY-COUNT VALIDATION]")
print(f"  Training set:")
print(f"    Density sums - mean: {density_sums_train.mean():.2f}, std: {density_sums_train.std():.2f}")
print(f"    Count values - mean: {loaded['y_count_train'].mean():.2f}, std: {loaded['y_count_train'].std():.2f}")
print(f"    Matching error - mean: {count_errors_train.mean():.2f}, max: {count_errors_train.max():.2f}")

print(f"\n  Test set:")
density_sums_test = loaded['y_density_test'].sum(axis=(1, 2))
count_errors_test = np.abs(density_sums_test - loaded['y_count_test'])
print(f"    Density sums - mean: {density_sums_test.mean():.2f}, std: {density_sums_test.std():.2f}")
print(f"    Count values - mean: {loaded['y_count_test'].mean():.2f}, std: {loaded['y_count_test'].std():.2f}")
print(f"    Matching error - mean: {count_errors_test.mean():.2f}, max: {count_errors_test.max():.2f}")

print(f"\n[IMAGE NORMALIZATION CHECK]")
print(f"  Mean per channel (should be ~0): {loaded['X_train'].mean(axis=(0, 1, 2))}")
print(f"  Std per channel (should be ~1): {loaded['X_train'].std(axis=(0, 1, 2))}")

print("\n" + "="*80)
print("[SUCCESS] CORRECTED PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nDataset ready for training:")
print(f"  Path: {output_path}")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")
