"""
Train a custom CNN from scratch on PlantVillage.

Architecture: 4 convolutional blocks + dense classifier.
Expected trainable params: ~2 M.

Usage:
    python train_cnn.py --source directory --data-dir ../data/PlantVillage --epochs 50
    python train_cnn.py --source tfds --epochs 50
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from ml.preprocess import get_dataset, IMG_SIZE, NUM_CLASSES
from ml.utils import (
    get_callbacks,
    compute_metrics,
    plot_training_history,
    plot_confusion_matrix,
    save_training_history,
    print_metrics_summary,
)

MODEL_NAME = "custom_cnn"


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_custom_cnn(num_classes: int, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """
    Build a 4-block CNN with batch normalisation and dropout.

    Block pattern: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # Block 1 — 32 filters (strides=2 for CPU memory efficiency)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2 — 64 filters
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3 — 128 filters
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 4 — 256 filters
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="CustomCNN")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(source: str = "directory", data_dir: str | None = None, epochs: int = 50):
    """End-to-end training pipeline for the custom CNN."""
    print(f"\n{'='*60}")
    print(f"  Training Custom CNN")
    print(f"{'='*60}\n")

    # 1. Load data
    train_ds, val_ds, test_ds, class_names, num_classes = get_dataset(
        source=source, data_dir=data_dir
    )

    # 2. Build model
    model = build_custom_cnn(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 3. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(MODEL_NAME),
    )

    # 4. Evaluate
    print("\nEvaluating on test set...")
    metrics = compute_metrics(model, test_ds, class_names, MODEL_NAME)
    print_metrics_summary(metrics)

    # 5. Plots
    plot_training_history(history, MODEL_NAME)
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names, MODEL_NAME
    )
    save_training_history(history, MODEL_NAME)

    return model, history, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Custom CNN on PlantVillage")
    parser.add_argument("--source", default="directory", choices=["directory", "tfds"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train(source=args.source, data_dir=args.data_dir, epochs=args.epochs)
