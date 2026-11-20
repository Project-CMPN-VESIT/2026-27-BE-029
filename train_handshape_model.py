import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def train():
    npz_path = "handshape_dataset.npz"
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")

    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_dim = X.shape[1]

    # Fast Instant Frame Classifier
    model = Sequential([
        Input(shape=(feature_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ModelCheckpoint('handshape_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    print("\nTraining instant hand shape model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal Instant Classifier Validation Accuracy: {val_acc * 100:.2f}%")
    print("Model saved as handshape_model.h5!")

if __name__ == "__main__":
    train()
