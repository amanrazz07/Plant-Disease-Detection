"""
Model loading and inference engine.

Manages multiple TensorFlow / Keras models and exposes a unified
``predict(image, model_name)`` interface for the API layer.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from io import BytesIO

from app.core.config import settings


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "mobilenetv2": "MobileNetV2",
    "resnet50": "ResNet50",
}

MODEL_DESCRIPTIONS = {
    "custom_cnn": "4-block CNN trained from scratch on PlantVillage (~2 M params)",
    "mobilenetv2": "MobileNetV2 with transfer learning from ImageNet (~3.5 M fine-tuned params)",
    "resnet50": "ResNet50 with transfer learning from ImageNet (~25.6 M fine-tuned params)",
}


class ModelManager:
    """Singleton-style manager that lazy-loads models on demand."""

    def __init__(self):
        self._models: dict[str, tf.keras.Model] = {}
        self._class_names: list[str] = []
        self._load_class_names()

    # ── Class names ───────────────────────────────────────────
    def _load_class_names(self):
        path = settings.MODELS_DIR / "class_names.json"
        if path.exists():
            with open(path) as f:
                self._class_names = json.load(f)
        else:
            # Fallback — import from preprocessing module
            try:
                from ml.preprocess import CLASS_NAMES
                self._class_names = CLASS_NAMES
            except ImportError:
                self._class_names = []

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    # ── Model loading ─────────────────────────────────────────
    def _model_path(self, model_name: str) -> Path:
        return settings.MODELS_DIR / f"{model_name}.keras"

    def is_available(self, model_name: str) -> bool:
        return self._model_path(model_name).exists()

    def load_model(self, model_name: str) -> tf.keras.Model:
        """Load a model from disk (cached after first load)."""
        if model_name in self._models:
            return self._models[model_name]

        path = self._model_path(model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {path}. "
                "Train it first with the corresponding train_*.py script."
            )

        print(f"Loading model '{model_name}' from {path} ...")
        model = tf.keras.models.load_model(str(path))
        self._models[model_name] = model
        print(f"  [OK] {model_name} loaded ({model.count_params():,} params)")
        return model

    def load_all_available(self):
        """Pre-load every model file found on disk."""
        for name in settings.AVAILABLE_MODELS:
            if self.is_available(name):
                self.load_model(name)

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    # ── Inference ─────────────────────────────────────────────
    def preprocess_image(self, image_bytes: bytes, target_size: int = settings.IMG_SIZE) -> np.ndarray:
        """Convert raw bytes to a (1, target_size, target_size, 3) float32 array in [0, 1]."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # add batch dim

    def predict(self, image_bytes: bytes, model_name: str | None = None):
        """
        Run inference on a single image.

        Returns
        -------
        dict with keys:
            prediction  — top-1 result dict
            top_5       — list of top-5 result dicts
            model_used  — name of the model
            inference_ms — time in milliseconds
        """
        model_name = model_name or settings.DEFAULT_MODEL
        model = self.load_model(model_name)
        target_size = settings.IMG_SIZE
        try:
            if hasattr(model, "input_shape") and model.input_shape and model.input_shape[1]:
                target_size = int(model.input_shape[1])
        except Exception:
            pass

        img = self.preprocess_image(image_bytes, target_size=target_size)

        start = time.perf_counter()
        probs = model.predict(img, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        top_indices = np.argsort(probs)[::-1][:5]
        top_5 = []
        for idx in top_indices:
            raw_name = self._class_names[idx]
            info = self._format_class(raw_name)
            info["confidence"] = float(probs[idx])
            top_5.append(info)

        from app.core.remedies import get_remedy
        top_prediction = top_5[0]
        remedy_info = get_remedy(top_prediction["class_name"])

        return {
            "prediction": top_prediction,
            "remedy": remedy_info,
            "top_5": top_5,
            "model_used": model_name,
            "inference_ms": round(elapsed_ms, 2),
        }

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _format_class(raw: str) -> dict:
        parts = raw.split("___")
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
        return {
            "class_name": raw,
            "plant": plant,
            "condition": condition.title(),
            "is_healthy": condition.lower() == "healthy",
        }

    def get_model_info(self, name: str) -> dict:
        """Return metadata dict for a single model."""
        available = self.is_available(name)
        info = {
            "name": name,
            "display_name": MODEL_DISPLAY_NAMES.get(name, name),
            "description": MODEL_DESCRIPTIONS.get(name, ""),
            "available": available,
            "size_mb": None,
            "accuracy": None,
        }
        if available:
            size = self._model_path(name).stat().st_size / (1024 * 1024)
            info["size_mb"] = round(size, 2)
            # Try to load stored accuracy
            metrics_path = settings.RESULTS_DIR / f"{name}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    info["accuracy"] = json.load(f).get("accuracy")
        return info


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
model_manager = ModelManager()
