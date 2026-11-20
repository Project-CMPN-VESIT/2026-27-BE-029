import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def augment_data(X, y):
    print(f"Applying data augmentation to {len(X)} training samples...", flush=True)
    augmented_X = [X]
    augmented_y = [y]

    # Augmentation 1: Subtle Gaussian noise on landmark features
    noise = np.random.normal(0, 0.012, X.shape).astype(np.float32)
    # Don't add noise to presence flags (indices 126 and 127 in base 135)
    noise[:, 126:128] = 0.0
    X_noisy = X + noise
    augmented_X.append(X_noisy)
    augmented_y.append(y)

    # Augmentation 2: Random slight scaling (0.97 to 1.03)
    scales = np.random.uniform(0.97, 1.03, size=(len(X), 1)).astype(np.float32)
    X_scaled = X.copy()
    X_scaled[:, :126] *= scales
    augmented_X.append(X_scaled)
    augmented_y.append(y)

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    print(f"Augmented training set shape: X={X_aug.shape}, y={y_aug.shape}", flush=True)
    return X_aug, y_aug

def train():
    npz_path = "words_100_dataset.npz"
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.", flush=True)
        return

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    num_classes = len(np.unique(y))
    print(f"Loaded dataset: X shape: {X.shape}, y shape: {y.shape}, classes: {num_classes}", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Augment training set only (validation remains clean untouched ground truth)
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    feature_dim = X.shape[1]

    # Deep high-accuracy Classifier for 100 classes
    model = Sequential([
        Input(shape=(feature_dim,)),
        Dense(384, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ModelCheckpoint('words_100_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    print("\nTraining high-accuracy 100-word model...", flush=True)
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=70,
        batch_size=64,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal 100-Word Classifier Validation Accuracy: {val_acc * 100:.2f}%", flush=True)
    print("Best model checkpoint saved as words_100_model.h5!", flush=True)

if __name__ == "__main__":
    train()
