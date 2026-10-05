"""
Shared utilities for training, evaluation, and visualisation.

Provides helpers for:
  - Training callbacks (early stopping, checkpointing, LR scheduling)
  - Metric computation (precision, recall, F1, confusion matrix)
  - Plot generation (training curves, confusion matrices)
  - Model metadata persistence
  - Inference-speed measurement
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"


# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------
def get_callbacks(model_name: str, patience: int = 10):
    """Return a standard set of Keras callbacks for training."""
    model_path = SAVED_MODELS_DIR / f"{model_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------
def compute_metrics(model, test_ds, class_names: list[str], model_name: str):
    """
    Evaluate *model* on *test_ds* and return a metrics dict.

    Also saves a JSON file under ``results/<model_name>_metrics.json``.
    """
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # Inference speed
    inference_ms = measure_inference_speed(model)

    # Model size on disk
    model_path = SAVED_MODELS_DIR / f"{model_name}.keras"
    model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0

    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "inference_ms": float(inference_ms),
        "model_size_mb": round(model_size_mb, 2),
        "total_params": int(model.count_params()),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {out}")

    return metrics


# ---------------------------------------------------------------------------
# Inference Speed
# ---------------------------------------------------------------------------
def measure_inference_speed(model, n_runs: int = 50):
    """Return average inference time in milliseconds for a single image."""
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    # Warm-up
    for _ in range(5):
        model.predict(dummy, verbose=0)
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
    return round(np.mean(times), 2)


# ---------------------------------------------------------------------------
# Plotting — Training History
# ---------------------------------------------------------------------------
def plot_training_history(history, model_name: str):
    """Save accuracy and loss curves for a training run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title(f"{model_name} — Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title(f"{model_name} — Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / f"{model_name}_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved → {path}")


# ---------------------------------------------------------------------------
# Plotting — Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names: list[str], model_name: str):
    """Save a heatmap of the confusion matrix."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# History Persistence
# ---------------------------------------------------------------------------
def save_training_history(history, model_name: str):
    """Serialise Keras ``History`` object to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    path = RESULTS_DIR / f"{model_name}_history.json"
    with open(path, "w") as f:
        json.dump(h, f, indent=2)
    print(f"Training history saved → {path}")


# ---------------------------------------------------------------------------
# Human-Readable Class Name
# ---------------------------------------------------------------------------
def format_class_name(raw: str) -> dict:
    """
    Convert ``'Tomato___Early_blight'`` into a structured dict:
    ``{'plant': 'Tomato', 'condition': 'Early Blight', 'is_healthy': False}``
    """
    parts = raw.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    is_healthy = condition.lower() == "healthy"
    return {
        "plant": plant,
        "condition": condition.title(),
        "is_healthy": is_healthy,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Print Summary
# ---------------------------------------------------------------------------
def print_metrics_summary(metrics: dict):
    """Pretty-print key evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  Model: {metrics['model_name']}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  Inference : {metrics['inference_ms']:.2f} ms")
    print(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
    print(f"  Parameters: {metrics['total_params']:,}")
    print(f"{'='*50}\n")
