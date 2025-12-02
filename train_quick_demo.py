"""
CSRNet Quick Training (20 epochs instead of 100)
For demonstration with accurate MSE/RMSE metrics
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("[CSRNet QUICK TRAINING - DEMO]")
print("="*80)

# Load corrected dataset
dataset_path = r"E:\DeepVision\processed_dataset_corrected\part_A_corrected.pkl"

print("\n[LOADING DATASET]")
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_density_train = data['y_density_train']
y_density_test = data['y_density_test']
y_count_train = data['y_count_train']
y_count_test = data['y_count_test']

print(f"[OK] Loaded {len(X_train)} train, {len(X_test)} test samples")

# Create model
print("\n[BUILDING MODEL]")
model = models.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=1),
    
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu', dilation_rate=2),
    
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    
    layers.Conv2D(1, (1, 1), padding='same', activation='relu'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"[OK] Model: {model.count_params():,} parameters")

# Setup
model_dir = r"E:\DeepVision\models\csrnet_quick_demo"
results_dir = r"E:\DeepVision\results\csrnet_quick_demo"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

y_density_train = np.expand_dims(y_density_train, axis=-1).astype(np.float32)
y_density_test = np.expand_dims(y_density_test, axis=-1).astype(np.float32)

# Train (20 epochs only - fast)
print("\n[TRAINING - 20 EPOCHS (Quick Demo)]")
print(f"[INFO] Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

history = model.fit(
    X_train, y_density_train,
    batch_size=16,
    epochs=20,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, verbose=0)
    ],
    verbose=1,
    shuffle=True
)

print(f"[OK] End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Evaluate
print("\n[EVALUATION]")
test_loss, test_mae = model.evaluate(X_test, y_density_test, verbose=0)

# Predictions
y_pred_test = model.predict(X_test, verbose=0)
pred_counts = np.sum(y_pred_test.reshape(len(y_pred_test), -1), axis=1)

# Metrics
mae_count = np.mean(np.abs(pred_counts - y_count_test))
rmse_count = np.sqrt(np.mean((pred_counts - y_count_test) ** 2))
errors = np.abs(pred_counts - y_count_test)

print(f"""
TEST RESULTS:
  Loss (MSE):       {test_loss:.6f}
  MAE (density):    {test_mae:.6f}
  
  COUNT PREDICTIONS (Primary):
  ✓ MAE:  {mae_count:.2f} people
  ✓ RMSE: {rmse_count:.2f} people
  
  Error stats:
  Min:  {errors.min():.2f}
  Max:  {errors.max():.2f}
  Median: {np.median(errors):.2f}
""")

# Save
model.save(os.path.join(model_dir, 'model.h5'))

results = {
    'mae_count': float(mae_count),
    'rmse_count': float(rmse_count),
    'test_loss': float(test_loss),
    'test_mae': float(test_mae),
    'epochs': len(history.history['loss']),
    'predictions': pred_counts.tolist(),
    'ground_truth': y_count_test.tolist(),
    'errors': errors.tolist()
}

with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0, 0].set_title('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['mae'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Val', linewidth=2)
axes[0, 1].set_title('MAE (Density)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_count_test, pred_counts, alpha=0.6)
min_v = min(y_count_test.min(), pred_counts.min())
max_v = max(y_count_test.max(), pred_counts.max())
axes[1, 0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
axes[1, 0].set_xlabel('True Count')
axes[1, 0].set_ylabel('Predicted Count')
axes[1, 0].set_title(f'Predictions vs GT (MAE={mae_count:.2f})')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Error (people)')
axes[1, 1].set_title(f'Error Distribution (RMSE={rmse_count:.2f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'results.png'), dpi=100, bbox_inches='tight')

print(f"\n[SAVED] Results: {os.path.join(results_dir, 'results.pkl')}")
print("="*80)
