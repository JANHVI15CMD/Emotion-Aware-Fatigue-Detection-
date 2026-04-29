import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Reshape,
                                     LSTM, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# 🔹 Load data
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Shape:", X.shape)
print("Label distribution:", np.bincount(y))

# ✅ FIX 1: Data is already normalized in create_spectrograms.py
# DO NOT re-normalize here — that was causing the prediction bug
X = np.nan_to_num(X)   # safety only

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Class weights
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# ✅ FIX 2: Data augmentation OUTSIDE the model (not inside Sequential)
# This prevents augmentation from affecting model.predict() at inference time
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(augment).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(16).prefetch(tf.data.AUTOTUNE)

# ✅ FIX 3: CNN-LSTM architecture (matches synopsis exactly)
# CNN extracts spatial features → Reshape → LSTM for temporal analysis
model = Sequential([
    Input(shape=(128, 128, 1)),

    # ── CNN Feature Extraction Block ──
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),           # → (64, 64, 32)

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),           # → (32, 32, 64)

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),           # → (16, 16, 128)

    # ── Reshape for LSTM: 16 time-steps × 2048 features ──
    Reshape((16, 16 * 128)),        # → (16, 2048)

    # ── LSTM Temporal Analysis Block ──
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),

    # ── Classification Head ──
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.summary()

# 🔹 Compile
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# 🔹 Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 🔹 Train
history = model.fit(
    train_ds,
    epochs=40,
    validation_data=test_ds,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 🔹 Evaluate
results = model.evaluate(test_ds)
print("\n✅ Test Accuracy  :", round(results[1], 4))
print("🎯 Precision      :", round(results[2], 4))
print("🎯 Recall         :", round(results[3], 4))

# 🔹 Save
model.save("cnn_model_best.keras")
print("\n🔥 Model saved as cnn_model_best.keras")