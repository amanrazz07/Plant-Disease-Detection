"""
Compare trained models and generate research-quality reports.

Reads ``results/<model>_metrics.json`` for each model and produces:
  - Bar charts for accuracy / precision / recall / F1
  - Training-curve overlays
  - Side-by-side confusion matrices
  - Summary CSV table

Usage:
    python compare_models.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAMES = ["custom_cnn", "mobilenetv2", "resnet50"]
DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenetv2": "MobileNetV2",
    "resnet50": "ResNet50",
}

# Color palette for each model
COLORS = {
    "custom_cnn": "#4ECDC4",
    "mobilenetv2": "#FF6B6B",
    "resnet50": "#45B7D1",
}


def load_metrics(model_name: str) -> dict | None:
    """Load metrics JSON for a model, return None if not found."""
    path = RESULTS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        print(f"  [!] Metrics not found for {model_name} -- skipping")
        return None
    with open(path) as f:
        return json.load(f)


def load_history(model_name: str) -> dict | None:
    """Load training history JSON for a model."""
    path = RESULTS_DIR / f"{model_name}_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_metrics_comparison(all_metrics: dict[str, dict]):
    """Bar chart comparing accuracy, precision, recall, F1 across models."""
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = list(all_metrics.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(n_models)
    bar_width = 0.5

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        values = [all_metrics[m][key] for m in models]
        colors = [COLORS.get(m, "#999") for m in models]
        bars = ax.bar(x, values, bar_width, color=colors, edgecolor="white", linewidth=1.5)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models], fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Model Comparison — Key Metrics", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "comparison_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Metrics comparison chart → {path}")


# ---------------------------------------------------------------------------
# Training Curve Overlay
# ---------------------------------------------------------------------------
def plot_training_curves_overlay(all_histories: dict[str, dict]):
    """Overlay training accuracy/loss curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, hist in all_histories.items():
        color = COLORS.get(name, "#999")
        display = DISPLAY_NAMES.get(name, name)
        epochs = range(1, len(hist["accuracy"]) + 1)

        ax1.plot(epochs, hist["accuracy"], label=f"{display} (train)", color=color, linewidth=2)
        ax1.plot(epochs, hist["val_accuracy"], label=f"{display} (val)",
                 color=color, linewidth=2, linestyle="--")

        ax2.plot(epochs, hist["loss"], label=f"{display} (train)", color=color, linewidth=2)
        ax2.plot(epochs, hist["val_loss"], label=f"{display} (val)",
                 color=color, linewidth=2, linestyle="--")

    ax1.set_title("Accuracy Curves", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — All Models", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "comparison_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Training curves overlay → {path}")


# ---------------------------------------------------------------------------
# Performance Summary (speed & size)
# ---------------------------------------------------------------------------
def plot_performance_comparison(all_metrics: dict[str, dict]):
    """Bar chart for inference speed and model size."""
    models = list(all_metrics.keys())
    n = len(models)
    displays = [DISPLAY_NAMES.get(m, m) for m in models]
    colors = [COLORS.get(m, "#999") for m in models]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(n)

    # Inference speed
    speeds = [all_metrics[m].get("inference_ms", 0) for m in models]
    bars1 = ax1.bar(x, speeds, 0.5, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars1, speeds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontweight="bold")
    ax1.set_title("Inference Speed (ms)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(displays)
    ax1.grid(axis="y", alpha=0.3)

    # Model size
    sizes = [all_metrics[m].get("model_size_mb", 0) for m in models]
    bars2 = ax2.bar(x, sizes, 0.5, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontweight="bold")
    ax2.set_title("Model Size (MB)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(displays)
    ax2.grid(axis="y", alpha=0.3)

    # Parameter count
    params = [all_metrics[m].get("total_params", 0) / 1e6 for m in models]
    bars3 = ax3.bar(x, params, 0.5, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars3, params):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}M", ha="center", fontweight="bold")
    ax3.set_title("Parameters (millions)", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(displays)
    ax3.grid(axis="y", alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "comparison_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Performance comparison → {path}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def export_summary_csv(all_metrics: dict[str, dict]):
    """Export a comparison table as CSV."""
    path = RESULTS_DIR / "comparison_summary.csv"
    fields = [
        "Model", "Accuracy", "Precision", "Recall", "F1 Score",
        "Inference (ms)", "Size (MB)", "Parameters",
    ]
    rows = []
    for name, m in all_metrics.items():
        rows.append([
            DISPLAY_NAMES.get(name, name),
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m.get('inference_ms', 0):.2f}",
            f"{m.get('model_size_mb', 0):.2f}",
            f"{m.get('total_params', 0):,}",
        ])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"  ✓ Summary CSV → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def compare():
    """Run the full comparison pipeline."""
    print(f"\n{'='*60}")
    print(f"  Model Comparison Report")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all available metrics
    all_metrics: dict[str, dict] = {}
    all_histories: dict[str, dict] = {}

    for name in MODEL_NAMES:
        m = load_metrics(name)
        if m is not None:
            all_metrics[name] = m
        h = load_history(name)
        if h is not None:
            all_histories[name] = h

    if not all_metrics:
        print("No model metrics found. Train at least one model first.")
        return

    print(f"\n  Found metrics for {len(all_metrics)} model(s): "
          f"{', '.join(DISPLAY_NAMES.get(n, n) for n in all_metrics)}\n")

    # Generate plots
    plot_metrics_comparison(all_metrics)
    if all_histories:
        plot_training_curves_overlay(all_histories)
    plot_performance_comparison(all_metrics)
    export_summary_csv(all_metrics)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Speed(ms)':>10} {'Size(MB)':>10}")
    print(f"{'-'*80}")
    for name, m in all_metrics.items():
        print(f"{DISPLAY_NAMES.get(name, name):<15} "
              f"{m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} "
              f"{m.get('inference_ms', 0):>10.2f} "
              f"{m.get('model_size_mb', 0):>10.2f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    compare()
