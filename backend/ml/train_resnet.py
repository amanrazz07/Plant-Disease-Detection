"""
Train ResNet50 via transfer learning on PlantVillage.

Strategy: freeze the ImageNet-pretrained base, train a new classifier head,
then optionally fine-tune the top layers.

Usage:
    python train_resnet.py --source directory --data-dir ../data/PlantVillage --epochs 30
    python train_resnet.py --source tfds --epochs 30
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

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

MODEL_NAME = "resnet50"


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_resnet50(num_classes: int, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """
    Build ResNet50 with a custom classification head.

    Phase 1 — freeze base, train head.
    Phase 2 — unfreeze top ~40 layers, fine-tune at low LR.
    """
    base = tf.keras.applications.ResNet50(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Phase 1: freeze

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # ResNet50 expects [0, 255] scaled to BGR zero-centered
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50_Transfer")
    return model, base


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    source: str = "directory",
    data_dir: str | None = None,
    epochs: int = 30,
    fine_tune_epochs: int = 15,
    fine_tune_at: int = 140,
):
    """Two-phase transfer-learning pipeline for ResNet50."""
    print(f"\n{'='*60}")
    print(f"  Training ResNet50 (Transfer Learning)")
    print(f"{'='*60}\n")

    # 1. Load data
    train_ds, val_ds, test_ds, class_names, num_classes = get_dataset(
        source=source, data_dir=data_dir
    )

    # 2. Build model
    model, base = build_resnet50(num_classes)
    model.summary()

    # ── Phase 1: Train classifier head ────────────────────────
    print("\n── Phase 1: Training classifier head (base frozen) ──\n")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(MODEL_NAME),
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────
    print(f"\n── Phase 2: Fine-tuning from layer {fine_tune_at} ──\n")
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_epochs = epochs + fine_tune_epochs
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_phase1.history["loss"]),
        epochs=total_epochs,
        callbacks=get_callbacks(MODEL_NAME, patience=7),
    )

    # Merge histories
    history_phase1.history = {
        k: history_phase1.history[k] + history_phase2.history[k]
        for k in history_phase1.history
    }

    # 3. Evaluate
    print("\nEvaluating on test set...")
    metrics = compute_metrics(model, test_ds, class_names, MODEL_NAME)
    print_metrics_summary(metrics)

    # 4. Plots
    plot_training_history(history_phase1, MODEL_NAME)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, MODEL_NAME)
    save_training_history(history_phase1, MODEL_NAME)

    return model, history_phase1, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet50 on PlantVillage")
    parser.add_argument("--source", default="directory", choices=["directory", "tfds"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-at", type=int, default=140,
                        help="Unfreeze base layers from this index onward")
    args = parser.parse_args()

    train(
        source=args.source,
        data_dir=args.data_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_at=args.fine_tune_at,
    )
