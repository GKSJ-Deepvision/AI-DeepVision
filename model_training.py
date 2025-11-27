"""
Deep Learning Model for Crowd Counting
- Uses CNN-based architecture
- Trains on processed dataset with density map targets
- Includes callbacks for monitoring and early stopping
- Provides evaluation metrics
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path
import json

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import tensorflow.keras.backend as K

# Custom loss function
def density_aware_mse(y_true, y_pred):
    """MSE loss weighted by density magnitude"""
    mse = K.mean(K.square(y_pred - y_true))
    return mse

# Custom metric: MAE on crowd counts
def count_mae(y_true, y_pred):
    """Mean Absolute Error on integrated counts"""
    true_counts = K.sum(K.sum(y_true, axis=-1), axis=-1)
    pred_counts = K.sum(K.sum(y_pred, axis=-1), axis=-1)
    return K.mean(K.abs(true_counts - pred_counts))

# ------------ CONFIGURATION -----------
CONFIG = {
    'processed_dir': r"E:\DeepVision\processed_dataset",
    'models_dir': r"E:\DeepVision\models",
    'results_dir': r"E:\DeepVision\results",
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.001,
    'img_size': (256, 256, 3),
    'density_size': (64, 64, 1)
}

os.makedirs(CONFIG['models_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

print("=" * 80)
print("🧠 CROWD COUNTING MODEL TRAINING")
print("=" * 80)

# ------------ LOAD DATASET -----------
print("\n📂 Loading processed dataset...")

dataset_path = os.path.join(CONFIG['processed_dir'], 'processed_dataset.pkl')
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_density_train = dataset['y_density_train']
y_density_test = dataset['y_density_test']
y_count_train = dataset['y_count_train']
y_count_test = dataset['y_count_test']

print(f"✅ Dataset loaded successfully")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_density_train: {y_density_train.shape}")
print(f"   y_density_test: {y_density_test.shape}")

# Add channel dimension to density maps
y_density_train = np.expand_dims(y_density_train, axis=-1)
y_density_test = np.expand_dims(y_density_test, axis=-1)

print(f"   y_density_train (with channel): {y_density_train.shape}")
print(f"   y_density_test (with channel): {y_density_test.shape}")

# ------------ BUILD MODEL -----------
print("\n" + "=" * 80)
print("🏗️  BUILDING MODEL ARCHITECTURE")
print("=" * 80)

def build_crowd_counting_model(input_shape, output_shape):
    """
    Build CNN-based model for crowd density estimation.
    
    Architecture:
    - Encoder: Progressive downsampling with feature extraction
    - Decoder: Progressive upsampling to generate density maps
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='image_input')
    
    # Encoder blocks with increasing filters
    print("\nEncoder Blocks:")
    
    # Block 1: 256 → 128
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    print(f"  Block 1: Output shape {x.shape[1:]}")
    
    # Block 2: 128 → 64
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    skip2 = x
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    print(f"  Block 2: Output shape {x.shape[1:]}")
    
    # Block 3: 64 → 32
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    skip3 = x
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    print(f"  Block 3: Output shape {x.shape[1:]}")
    
    # Block 4: 32 → 16 (Bottleneck)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    print(f"  Bottleneck: Output shape {x.shape[1:]}")
    
    # Decoder blocks with upsampling
    print("\nDecoder Blocks:")
    
    # Upsample 1: 16 → 32
    x = layers.UpSampling2D((2, 2), name='upsample1')(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='deconv1_1')(x)
    x = layers.BatchNormalization(name='bn_d1_1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='deconv1_2')(x)
    x = layers.BatchNormalization(name='bn_d1_2')(x)
    print(f"  Upsample 1: Output shape {x.shape[1:]}")
    
    # Upsample 2: 32 → 64
    x = layers.UpSampling2D((2, 2), name='upsample2')(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv2_1')(x)
    x = layers.BatchNormalization(name='bn_d2_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='deconv2_2')(x)
    x = layers.BatchNormalization(name='bn_d2_2')(x)
    print(f"  Upsample 2: Output shape {x.shape[1:]}")
    
    # Upsample 3: 64 → 128
    x = layers.UpSampling2D((2, 2), name='upsample3')(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='deconv3_1')(x)
    x = layers.BatchNormalization(name='bn_d3_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='deconv3_2')(x)
    x = layers.BatchNormalization(name='bn_d3_2')(x)
    print(f"  Upsample 3: Output shape {x.shape[1:]}")
    
    # Output layer
    # Note: we need to downsample from 256 to 64 for density map output
    x = layers.MaxPooling2D((4, 4), name='final_pool')(x)
    outputs = layers.Conv2D(1, (1, 1), activation='relu', padding='same', name='density_output')(x)
    print(f"  Output: {outputs.shape[1:]}")
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CrowdCounting')
    return model

# Build model
model = build_crowd_counting_model(CONFIG['img_size'], CONFIG['density_size'])

# Compile model
print("\n" + "=" * 80)
print("⚙️  COMPILING MODEL")
print("=" * 80)

optimizer = Adam(learning_rate=CONFIG['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss=MeanSquaredError(),
    metrics=[
        MeanAbsoluteError(name='mae'),
        count_mae
    ]
)

print("✅ Model compiled with:")
print(f"   Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"   Loss: Mean Squared Error")
print(f"   Metrics: MAE, Count MAE")

# Model summary
print("\n" + "-" * 80)
model.summary()
print("-" * 80)

# ------------ TRAINING -----------
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(CONFIG['models_dir'], 'best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

tensorboard = callbacks.TensorBoard(
    log_dir=os.path.join(CONFIG['results_dir'], 'logs'),
    histogram_freq=1,
    write_graph=True
)

# Train model
history = model.fit(
    X_train, y_density_train,
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['epochs'],
    validation_data=(X_test, y_density_test),
    callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard],
    verbose=1
)

# Save final model
final_model_path = os.path.join(CONFIG['models_dir'], 'final_model.h5')
model.save(final_model_path)
print(f"\n✅ Final model saved: {final_model_path}")

# Save model architecture
model_json_path = os.path.join(CONFIG['models_dir'], 'model_architecture.json')
with open(model_json_path, 'w') as f:
    f.write(model.to_json())

# ------------ EVALUATION -----------
print("\n" + "=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)

# Predictions
print("\n🔮 Making predictions on test set...")
y_pred_test = model.predict(X_test, batch_size=CONFIG['batch_size'], verbose=1)

# Calculate metrics
test_loss, test_mae, test_count_mae = model.evaluate(
    X_test, y_density_test,
    batch_size=CONFIG['batch_size'],
    verbose=0
)

print(f"\nTest Set Metrics:")
print(f"  Loss (MSE): {test_loss:.6f}")
print(f"  MAE: {test_mae:.6f}")
print(f"  Count MAE: {test_count_mae:.6f}")

# Count predictions
true_counts_test = np.sum(y_density_test.reshape(len(y_density_test), -1), axis=1)
pred_counts_test = np.sum(y_pred_test.reshape(len(y_pred_test), -1), axis=1)

mae_counts = np.mean(np.abs(true_counts_test - pred_counts_test))
rmse_counts = np.sqrt(np.mean((true_counts_test - pred_counts_test) ** 2))
mape_counts = np.mean(np.abs((true_counts_test - pred_counts_test) / (true_counts_test + 1)))

print(f"\nCount Prediction Metrics:")
print(f"  MAE: {mae_counts:.4f}")
print(f"  RMSE: {rmse_counts:.4f}")
print(f"  MAPE: {mape_counts:.4f}")

# ------------ SAVE RESULTS -----------
print("\n" + "=" * 80)
print("💾 SAVING RESULTS")
print("=" * 80)

# Save history
history_path = os.path.join(CONFIG['results_dir'], 'training_history.json')
history_data = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'mae': [float(x) for x in history.history['mae']],
    'val_mae': [float(x) for x in history.history['val_mae']],
    'count_mae': [float(x) for x in history.history['count_mae']],
    'val_count_mae': [float(x) for x in history.history['val_count_mae']]
}
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)
print(f"✅ Training history saved: {history_path}")

# Save evaluation metrics
metrics_path = os.path.join(CONFIG['results_dir'], 'evaluation_metrics.json')
metrics_data = {
    'test_loss': float(test_loss),
    'test_mae': float(test_mae),
    'test_count_mae': float(test_count_mae),
    'count_mae': float(mae_counts),
    'count_rmse': float(rmse_counts),
    'count_mape': float(mape_counts),
    'num_test_samples': len(X_test),
    'true_counts_stats': {
        'min': float(np.min(true_counts_test)),
        'max': float(np.max(true_counts_test)),
        'mean': float(np.mean(true_counts_test)),
        'std': float(np.std(true_counts_test))
    },
    'pred_counts_stats': {
        'min': float(np.min(pred_counts_test)),
        'max': float(np.max(pred_counts_test)),
        'mean': float(np.mean(pred_counts_test)),
        'std': float(np.std(pred_counts_test))
    }
}
with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=2)
print(f"✅ Evaluation metrics saved: {metrics_path}")

# ------------ PLOT RESULTS -----------
print("\n📈 Creating result plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(alpha=0.3)

# MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('MAE', fontsize=12)
axes[0, 1].set_title('Training and Validation MAE', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(alpha=0.3)

# Count MAE
axes[1, 0].plot(history.history['count_mae'], label='Training Count MAE', linewidth=2)
axes[1, 0].plot(history.history['val_count_mae'], label='Validation Count MAE', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Count MAE', fontsize=12)
axes[1, 0].set_title('Training and Validation Count MAE', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(alpha=0.3)

# True vs Predicted counts
axes[1, 1].scatter(true_counts_test, pred_counts_test, alpha=0.6, s=50)
min_val = min(true_counts_test.min(), pred_counts_test.min())
max_val = max(true_counts_test.max(), pred_counts_test.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('True Count', fontsize=12)
axes[1, 1].set_ylabel('Predicted Count', fontsize=12)
axes[1, 1].set_title('True vs Predicted Counts (Test Set)', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
result_plot_path = os.path.join(CONFIG['results_dir'], 'training_results.png')
fig.savefig(result_plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Results plot saved: {result_plot_path}")
plt.close()

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nResults directory: {CONFIG['results_dir']}")
print(f"Models directory: {CONFIG['models_dir']}")
print("=" * 80)
