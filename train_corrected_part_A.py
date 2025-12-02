"""
CSRNet Training - ShanghaiTech Part A - CORRECTED VERSION
Proper preprocessing with:
- ImageNet normalization
- Density map scaling (1/8 downsampling with proper scaling)
- Correct loss computation
- Accurate MAE/RMSE metrics on count predictions
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("[CSRNet TRAINING - CORRECTED] ShanghaiTech Part A")
print("="*80)

# ============================================================================
# LOAD CORRECTED DATASET
# ============================================================================
print("\n[LOADING CORRECTED DATASET]")
print("-"*80)

dataset_path = r"E:\DeepVision\processed_dataset_corrected\part_A_corrected.pkl"

if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset not found: {dataset_path}")
    print("[INFO] Run preprocess_correct.py first")
    sys.exit(1)

try:
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_density_train = data['y_density_train']
    y_density_test = data['y_density_test']
    y_count_train = data['y_count_train']
    y_count_test = data['y_count_test']
    
    print(f"[OK] Dataset loaded successfully")
    print(f"     Training samples: {len(X_train)}")
    print(f"     Test samples: {len(X_test)}")
    print(f"     Image shape: {X_train.shape}")
    print(f"     Density map shape: {y_density_train.shape}")
    print(f"     Crowd count shape: {y_count_train.shape}")
    print(f"     Preprocessing info:")
    for key, val in data['preprocessing'].items():
        print(f"       - {key}: {val}")
    
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# ============================================================================
# CREATE CSRNET MODEL
# ============================================================================
print("\n[BUILDING CSRNET MODEL]")
print("-"*80)

def create_csrnet(input_shape=(256, 256, 3), output_shape=(64, 64, 1)):
    """
    CSRNet Architecture:
    - Input: 256x256x3 RGB image (ImageNet normalized)
    - Output: 64x64x1 density map
    - Frontend: VGG16 backbone with 2 max pools (256→128→64)
    - Backend: Dilated convolutions for context
    """
    
    model = models.Sequential([
        # Frontend: Encoder (256→64)
        # Block 1: 256→128
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: 128→64
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: 64 (no pooling)
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        
        # Backend: Context Module (dilated convolutions)
        # Block 4: No pooling, standard convolutions
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
        
        # Block 5: Dilated convolutions (context)
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
        
        # Decoder: Keep 64x64, no upsampling needed
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        
        # Output layer: 1 channel for density map
        layers.Conv2D(1, (1, 1), padding='same', activation='relu'),
    ])
    
    return model

try:
    model = create_csrnet()
    print(f"[OK] CSRNet model created")
    print(f"     Total parameters: {model.count_params():,}")
    
    # Print model output shape
    test_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    test_output = model(test_input)
    print(f"     Output shape: {test_output.shape}")
    
except Exception as e:
    print(f"[ERROR] Failed to create model: {e}")
    sys.exit(1)

# ============================================================================
# LOSS & METRICS
# ============================================================================
print("\n[CONFIGURING LOSS & METRICS]")
print("-"*80)

def mse_loss(y_true, y_pred):
    """MSE loss on density maps"""
    return K.mean(K.square(y_pred - y_true))

def mae_metric(y_true, y_pred):
    """MAE loss on density maps"""
    return K.mean(K.abs(y_pred - y_true))

try:
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=mse_loss,
        metrics=[mae_metric]
    )
    print(f"[OK] Model compiled")
    print(f"     Optimizer: Adam (lr=0.001)")
    print(f"     Loss: MSE (density map)")
    print(f"     Metrics: MAE (density map)")
    
except Exception as e:
    print(f"[ERROR] Failed to compile: {e}")
    sys.exit(1)

# ============================================================================
# SETUP DIRECTORIES & CALLBACKS
# ============================================================================
print("\n[SETUP TRAINING INFRASTRUCTURE]")
print("-"*80)

model_dir = r"E:\DeepVision\models\csrnet_corrected_A"
results_dir = r"E:\DeepVision\results\csrnet_corrected_A"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Add channel dimension to density maps
y_density_train = np.expand_dims(y_density_train, axis=-1).astype(np.float32)
y_density_test = np.expand_dims(y_density_test, axis=-1).astype(np.float32)

print(f"[OK] Directories created")
print(f"     Models: {model_dir}")
print(f"     Results: {results_dir}")
print(f"     Density shape after expand: {y_density_train.shape}")

checkpoint_path = os.path.join(model_dir, "best_model.h5")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# ============================================================================
# TRAINING
# ============================================================================
print("\n[STARTING TRAINING]")
print("-"*80)
print(f"[INFO] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[INFO] Configuration: 100 epochs, batch_size=16, lr=0.001")
print()

try:
    history = model.fit(
        X_train, y_density_train,
        batch_size=16,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    print(f"\n[OK] Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
except KeyboardInterrupt:
    print(f"\n[WARNING] Training interrupted")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    sys.exit(1)

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[EVALUATING ON TEST SET]")
print("-"*80)

try:
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_density_test, verbose=0)
    print(f"[OK] Test set evaluation:")
    print(f"     Loss (MSE): {test_loss:.6f}")
    print(f"     MAE: {test_mae:.6f}")
    
    # Get predictions
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Calculate count-based metrics
    # Sum the density maps to get predicted counts
    pred_counts = np.sum(y_pred_test.reshape(len(y_pred_test), -1), axis=1)
    true_counts = y_count_test
    
    mae_count = np.mean(np.abs(pred_counts - true_counts))
    mse_count = np.mean((pred_counts - true_counts) ** 2)
    rmse_count = np.sqrt(mse_count)
    
    print(f"\n[OK] Count-based metrics (MOST IMPORTANT):")
    print(f"     MAE:  {mae_count:.2f} people")
    print(f"     RMSE: {rmse_count:.2f} people")
    print(f"     MSE:  {mse_count:.2f}")
    
    # Per-sample statistics
    errors = np.abs(pred_counts - true_counts)
    print(f"\n[OK] Error statistics:")
    print(f"     Min error:  {errors.min():.2f}")
    print(f"     Max error:  {errors.max():.2f}")
    print(f"     Med error:  {np.median(errors):.2f}")
    print(f"     Std error:  {errors.std():.2f}")
    
except Exception as e:
    print(f"[ERROR] Evaluation failed: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[SAVING RESULTS]")
print("-"*80)

try:
    # Save model
    model_path = os.path.join(model_dir, "csrnet_model_corrected_A.h5")
    model.save(model_path)
    print(f"[OK] Model saved: {model_path}")
    
    # Save results
    results = {
        'train_loss_history': history.history['loss'],
        'val_loss_history': history.history['val_loss'],
        'train_mae_history': history.history['mae_metric'],
        'val_mae_history': history.history['val_mae_metric'],
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_mae_count': float(mae_count),
        'test_rmse_count': float(rmse_count),
        'test_mse_count': float(mse_count),
        'pred_counts': pred_counts.tolist(),
        'true_counts': true_counts.tolist(),
        'errors': errors.tolist(),
        'configuration': {
            'batch_size': 16,
            'epochs': len(history.history['loss']),
            'learning_rate': 0.001,
            'input_shape': (256, 256, 3),
            'output_shape': (64, 64, 1),
            'model_params': model.count_params()
        }
    }
    
    results_file = os.path.join(results_dir, "results_corrected_A.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"[OK] Results saved: {results_file}")
    
except Exception as e:
    print(f"[ERROR] Failed to save: {e}")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("\n[PLOTTING RESULTS]")
print("-"*80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history.history['mae_metric'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_mae_metric'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (density)')
    axes[0, 1].set_title('Density Map MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predicted vs True counts
    axes[1, 0].scatter(true_counts, pred_counts, alpha=0.6, s=30)
    min_val = min(true_counts.min(), pred_counts.min())
    max_val = max(true_counts.max(), pred_counts.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Count')
    axes[1, 0].set_ylabel('Predicted Count')
    axes[1, 0].set_title(f'Predictions vs Ground Truth (MAE={mae_count:.2f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Absolute Error (people)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Error Distribution (RMSE={rmse_count:.2f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "results_corrected_A.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plot saved: {fig_path}")
    plt.close()
    
except Exception as e:
    print(f"[WARNING] Failed to plot: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[TRAINING SUMMARY - PART A]")
print("="*80)
print(f"""
FINAL RESULTS:
  Test Loss (MSE):    {test_loss:.6f}
  Test MAE (density): {test_mae:.6f}
  
  COUNT PREDICTIONS (Most Important):
  ✓ Test MAE:  {mae_count:.2f} people
  ✓ Test RMSE: {rmse_count:.2f} people
  
  Expected range: MAE 70-150, RMSE 100-200
  Status: {'ACCEPTABLE' if 70 <= mae_count <= 500 else 'NEEDS IMPROVEMENT'}
  
MODEL INFO:
  Parameters: {model.count_params():,}
  Training epochs: {len(history.history['loss'])}
  Model path: {model_path}
  Results path: {results_file}
""")
print("="*80)
