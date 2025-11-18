"""
Advanced Visualization Pipeline
- Visualizes original images with crowd annotations
- Displays density maps
- Compares original vs density map side-by-side
- Generates statistical analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import pickle
from pathlib import Path
import cv2

# ------------ CONFIGURATION & PATHS -------------
CONFIG = {
    'dataset_root': r"E:\DeepVision\ShanghaiTech\part_A",
    'processed_dir': r"E:\DeepVision\processed_dataset",
    'visualizations_dir': r"E:\DeepVision\visualizations"
}

os.makedirs(CONFIG['visualizations_dir'], exist_ok=True)

print("=" * 70)
print("📊 ADVANCED VISUALIZATION PIPELINE")
print("=" * 70)

# ------------ LOAD PROCESSED DATASET -------------
print("\n🔄 Loading processed dataset...")

dataset_path = os.path.join(CONFIG['processed_dir'], 'processed_dataset.pkl')
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_density_train = dataset['y_density_train']
y_density_test = dataset['y_density_test']
y_count_train = dataset['y_count_train']
y_count_test = dataset['y_count_test']
all_samples = dataset['all_samples']

print(f"✅ Loaded {len(all_samples)} total samples")
print(f"   Training: {len(X_train)}, Testing: {len(X_test)}")

# ------------ VISUALIZATION 1: ORIGINAL IMAGES & DENSITY MAPS (SIDE-BY-SIDE) -----------
print("\n🎨 Creating Visualization 1: Original Images vs Density Maps...")

num_samples = 12
fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))

for idx in range(num_samples):
    sample = all_samples[idx]
    img = sample['image']
    density = sample['density']
    count = sample['count']
    filename = sample['filename']
    
    # Original image
    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title(f"{filename}\nCount: {count}", fontsize=10, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Density map
    im = axes[idx, 1].imshow(density, cmap='hot', interpolation='bilinear')
    axes[idx, 1].set_title(f"Density Map\nMax: {density.max():.2f}", fontsize=10)
    axes[idx, 1].axis('off')
    plt.colorbar(im, ax=axes[idx, 1], label='Density')
    
    # Annotated image with crowd points
    img_annotated = img.copy()
    points = sample['points']
    
    # Scale points to original size
    orig_h, orig_w = sample['original_shape']
    scale_y = orig_h / img.shape[0]
    scale_x = orig_w / img.shape[1]
    
    for point in points:
        x, y = int(point[0] / scale_x), int(point[1] / scale_y)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_annotated, (x, y), 3, (0, 1, 0), -1)
    
    axes[idx, 2].imshow(img_annotated)
    axes[idx, 2].set_title(f"Annotated ({len(points)} points)", fontsize=10)
    axes[idx, 2].axis('off')

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['visualizations_dir'], '01_images_vs_density_maps.png'), dpi=150, bbox_inches='tight')
print("✅ Saved: 01_images_vs_density_maps.png")
plt.close()

# ------------ VISUALIZATION 2: CROWD COUNT DISTRIBUTION -----------
print("\n📈 Creating Visualization 2: Crowd Count Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(y_count_train, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
axes[0, 0].hist(y_count_test, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
axes[0, 0].set_xlabel('Crowd Count', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram of Crowd Counts', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Box plot
data_to_plot = [y_count_train, y_count_test]
bp = axes[0, 1].boxplot(data_to_plot, tick_labels=['Train', 'Test'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
axes[0, 1].set_ylabel('Crowd Count', fontsize=12)
axes[0, 1].set_title('Box Plot of Crowd Counts', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Statistics table
stats_text = f"""
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {len(all_samples)}
Training Samples: {len(X_train)}
Testing Samples: {len(X_test)}

Crowd Count Range:
Min: {np.min(y_count_train):.0f}  |  Max: {np.max(y_count_train):.0f}

Mean Count:
Train: {np.mean(y_count_train):.2f}
Test: {np.mean(y_count_test):.2f}

Std Dev:
Train: {np.std(y_count_train):.2f}
Test: {np.std(y_count_test):.2f}
"""
axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].axis('off')

# Cumulative distribution
sorted_counts = np.sort(y_count_train)
cumsum = np.cumsum(np.ones_like(sorted_counts)) / len(sorted_counts)
axes[1, 1].plot(sorted_counts, cumsum, linewidth=2, color='blue', label='Train')
sorted_counts_test = np.sort(y_count_test)
cumsum_test = np.cumsum(np.ones_like(sorted_counts_test)) / len(sorted_counts_test)
axes[1, 1].plot(sorted_counts_test, cumsum_test, linewidth=2, color='red', label='Test')
axes[1, 1].set_xlabel('Crowd Count', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 1].set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['visualizations_dir'], '02_crowd_count_analysis.png'), dpi=150, bbox_inches='tight')
print("✅ Saved: 02_crowd_count_analysis.png")
plt.close()

# ------------ VISUALIZATION 3: DENSITY MAP INTENSITY ANALYSIS -----------
print("\n🔥 Creating Visualization 3: Density Map Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Density map value distribution
all_densities = np.concatenate([y_density_train, y_density_test])
densities_flat = all_densities.flatten()

axes[0, 0].hist(densities_flat[densities_flat > 0], bins=100, color='orange', edgecolor='black')
axes[0, 0].set_xlabel('Density Value', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Density Map Values', fontsize=13, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(alpha=0.3)

# Density map sum vs crowd count
density_sums_train = np.sum(y_density_train.reshape(len(y_density_train), -1), axis=1)
axes[0, 1].scatter(y_count_train, density_sums_train, alpha=0.6, s=50, color='blue', label='Train')
density_sums_test = np.sum(y_density_test.reshape(len(y_density_test), -1), axis=1)
axes[0, 1].scatter(y_count_test, density_sums_test, alpha=0.6, s=50, color='red', label='Test')
axes[0, 1].set_xlabel('Crowd Count', fontsize=12)
axes[0, 1].set_ylabel('Density Map Sum', fontsize=12)
axes[0, 1].set_title('Correlation: Crowd Count vs Density Sum', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Max density per sample
max_densities_train = np.max(y_density_train.reshape(len(y_density_train), -1), axis=1)
max_densities_test = np.max(y_density_test.reshape(len(y_density_test), -1), axis=1)

axes[1, 0].scatter(y_count_train, max_densities_train, alpha=0.6, s=50, color='blue', label='Train')
axes[1, 0].scatter(y_count_test, max_densities_test, alpha=0.6, s=50, color='red', label='Test')
axes[1, 0].set_xlabel('Crowd Count', fontsize=12)
axes[1, 0].set_ylabel('Max Density Value', fontsize=12)
axes[1, 0].set_title('Correlation: Crowd Count vs Max Density', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Sample density maps (gallery)
axes[1, 1].axis('off')
info_text = f"""
Density Map Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape per sample: {y_density_train[0].shape}
Num. of samples: {len(all_densities)}

Density Value Range:
Min: {np.min(all_densities):.6f}
Max: {np.max(all_densities):.6f}
Mean: {np.mean(all_densities):.6f}
Median: {np.median(all_densities):.6f}

Non-zero Values:
Count: {np.sum(all_densities > 0)}
Percentage: {100*np.sum(all_densities > 0)/all_densities.size:.2f}%

Correlation (Count vs Density Sum):
r = {np.corrcoef(np.concatenate([y_count_train, y_count_test]), 
                  np.concatenate([density_sums_train, density_sums_test]))[0, 1]:.4f}
"""
axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['visualizations_dir'], '03_density_map_analysis.png'), dpi=150, bbox_inches='tight')
print("✅ Saved: 03_density_map_analysis.png")
plt.close()

# ------------ VISUALIZATION 4: SAMPLE GALLERY -----------
print("\n🖼️ Creating Visualization 4: Sample Gallery...")

num_samples_to_show = 16
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for idx in range(num_samples_to_show):
    sample = all_samples[idx]
    img = sample['image']
    density = sample['density']
    count = sample['count']
    
    # Create composite: image + density map overlay
    img_display = img.copy()
    
    # Normalize density map for overlay
    if density.max() > 0:
        density_norm = density / density.max()
    else:
        density_norm = density
    
    # Overlay density map on image (scaled to image size)
    density_overlay = cv2.resize(density_norm, (img.shape[1], img.shape[0]))
    density_color = plt.cm.hot(density_overlay)[:, :, :3]
    
    # Blend image with density map
    alpha = 0.5
    img_blended = (1 - alpha) * img_display + alpha * density_color
    
    axes[idx].imshow(img_blended)
    axes[idx].set_title(f"{sample['filename']}\nCount: {count}", fontsize=10, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
fig.savefig(os.path.join(CONFIG['visualizations_dir'], '04_sample_gallery.png'), dpi=150, bbox_inches='tight')
print("✅ Saved: 04_sample_gallery.png")
plt.close()

print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS COMPLETED!")
print("=" * 70)
print(f"\nVisualizations saved in: {CONFIG['visualizations_dir']}")
print("\nGenerated files:")
print("  1. 01_images_vs_density_maps.png - Side-by-side comparison")
print("  2. 02_crowd_count_analysis.png - Statistical analysis")
print("  3. 03_density_map_analysis.png - Density map properties")
print("  4. 04_sample_gallery.png - Sample gallery with overlays")
print("=" * 70)
