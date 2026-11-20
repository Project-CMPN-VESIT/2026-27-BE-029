import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def train():
    npz_path = "dataset.npz"
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Run extraction first!")
        return

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")

    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sequence_length = X.shape[1]
    feature_dim = X.shape[2]

    # Build sequence classification model
    model = Sequential([
        Input(shape=(sequence_length, feature_dim)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
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
        ModelCheckpoint('isl_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

    # Save to TFLite as well for ultra-fast CPU inference
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable TF select ops for LSTM support in TFLite if needed
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    try:
        tflite_model = converter.convert()
        with open('isl_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Model also saved to isl_model.tflite")
    except Exception as e:
        print("TFLite conversion note (H5 model is ready):", e)

if __name__ == "__main__":
    train()
