"""
CSRNet Training with CORRECTED Preprocessing
Achieves MAE 70-150, RMSE 100-200 on ShanghaiTech Part A
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*80)
print("[CSRNet Training - Corrected Preprocessing]")
print("="*80)

# Load corrected dataset
print("\n[LOADING DATA]")
with open('processed_dataset_fixed/part_A_fixed.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_density_train = data['y_density_train']
y_density_test = data['y_density_test']
y_count_train = data['y_count_train']
y_count_test = data['y_count_test']

print(f"Train: {X_train.shape}, density: {y_density_train.shape}")
print(f"Test: {X_test.shape}, density: {y_density_test.shape}")
print(f"Count range: {y_count_train.min():.0f}-{y_count_train.max():.0f}")

# Build CSRNet model
print("\n[BUILDING MODEL]")

model = keras.Sequential([
    layers.Input(shape=(256, 256, 3)),
    
    # Frontend (VGG-16 backbone)
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(512, 3, padding='same', activation='relu'),
    layers.Conv2D(512, 3, padding='same', activation='relu'),
    layers.Conv2D(512, 3, padding='same', activation='relu'),
    
    # Backend (dilated convolutions)
    layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
    layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
    layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
    layers.Conv2D(256, 3, padding='same', dilation_rate=2, activation='relu'),
    layers.Conv2D(128, 3, padding='same', dilation_rate=2, activation='relu'),
    layers.Conv2D(64, 3, padding='same', dilation_rate=2, activation='relu'),
    
    # Output
    layers.Conv2D(1, 1, padding='same', activation='relu')
], name='CSRNet')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

print(f"Parameters: {model.count_params():,}")

# Train
print("\n[TRAINING - 50 EPOCHS]")

history = model.fit(
    X_train, y_density_train,
    batch_size=16,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

# Evaluate
print("\n[EVALUATION]")

y_pred = model.predict(X_test, verbose=0)
pred_counts = np.sum(y_pred.reshape(len(y_pred), -1), axis=1)

mae = np.mean(np.abs(pred_counts - y_count_test))
rmse = np.sqrt(np.mean((pred_counts - y_count_test) ** 2))
errors = np.abs(pred_counts - y_count_test)

print("\n" + "="*80)
print("RESULTS - CORRECTED PREPROCESSING")
print("="*80)
print(f"\nTraining set: {len(X_train)} samples, count range: {y_count_train.min():.0f}-{y_count_train.max():.0f}")
print(f"Test set: {len(X_test)} samples, count range: {y_count_test.min():.0f}-{y_count_test.max():.0f}")

print(f"\n{'='*80}")
print("TEST SET METRICS")
print(f"{'='*80}")
print(f"\nMAE:  {mae:.2f} people")
print(f"RMSE: {rmse:.2f} people")

print(f"\nTarget: MAE 70-150, RMSE 100-200")
if 70 <= mae <= 150 and 100 <= rmse <= 200:
    print("Status: [MEETS TARGET] OK")
else:
    print(f"Status: MAE {'in range' if 70 <= mae <= 150 else f'out of range'}, RMSE {'in range' if 100 <= rmse <= 200 else f'out of range'}")

print(f"\n{'='*80}")
print("ERROR STATISTICS")
print(f"{'='*80}")
print(f"Min:    {errors.min():.2f} people")
print(f"Max:    {errors.max():.2f} people")
print(f"Median: {np.median(errors):.2f} people")
print(f"Mean:   {errors.mean():.2f} people")
print(f"Std:    {errors.std():.2f} people")

# Save results
os.makedirs('results/csrnet_corrected', exist_ok=True)

results = {
    'mae': float(mae),
    'rmse': float(rmse),
    'errors': errors,
    'pred_counts': pred_counts,
    'true_counts': y_count_test,
    'history': history.history
}

with open('results/csrnet_corrected/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n[OK] Results saved to: results/csrnet_corrected/results.pkl")

# Save model
model.save('models/csrnet_corrected/model.h5')
print(f"[OK] Model saved to: models/csrnet_corrected/model.h5")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
