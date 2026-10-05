"""
Data preprocessing pipeline for Plant Disease Detection.
Handles loading PlantVillage dataset, preprocessing, augmentation, and splitting.

Uses PIL-based on-the-fly loading wrapped in tf.py_function to avoid both:
1. Windows file-handle exhaustion (error 1450) from tf.data C++ filesystem
2. High RAM allocation crashes from storing 50K+ images in memory at once

Usage:
    python preprocess.py --source directory --data-dir ../data/PlantVillage
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

# Default max samples per class for CPU training speed.
# Can be overridden via MAX_SAMPLES_PER_CLASS env var (set to 0 or 'all' for full dataset)
_ENV_MAX_SAMPLES = os.environ.get("MAX_SAMPLES_PER_CLASS", "100")
try:
    DEFAULT_MAX_SAMPLES = int(_ENV_MAX_SAMPLES) if _ENV_MAX_SAMPLES.lower() not in ("0", "none", "all") else None
except ValueError:
    DEFAULT_MAX_SAMPLES = 100

# PlantVillage 38 class names
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)  # 38


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------
def create_augmentation_layer():
    """Create a Keras data-augmentation sequential layer."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


# ---------------------------------------------------------------------------
# Safe PIL Loader (Windows compatible, streaming batch-by-batch)
# ---------------------------------------------------------------------------
def _load_image_np(path_tensor):
    path_str = path_tensor.numpy().decode("utf-8")
    try:
        with Image.open(path_str) as img:
            img = img.convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


def _py_func_mapper(path, label):
    [image] = tf.py_function(_load_image_np, [path], [tf.float32])
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    label.set_shape(())
    return image, label


def _build_tf_dataset(paths, labels, is_train=False):
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_train:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), seed=SEED)
    ds = ds.map(_py_func_mapper, num_parallel_calls=2)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(1)
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_dataset(
    source: str = "directory",
    data_dir: str | Path | None = None,
    max_samples_per_class: int | None = DEFAULT_MAX_SAMPLES,
):
    """
    Load and prepare the PlantVillage dataset.

    Returns train_ds, val_ds, test_ds, class_names, num_classes
    where each *_ds is a batched tf.data.Dataset.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Download PlantVillage from Kaggle and extract into that path."
        )

    # Discover class folders (sorted for reproducibility)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls_dir in class_dirs:
        idx = class_to_idx[cls_dir.name]
        cls_imgs = []
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                cls_imgs.append(str(f))
        
        # Balance / sample if requested
        if max_samples_per_class and max_samples_per_class > 0:
            cls_imgs = cls_imgs[:max_samples_per_class]

        all_paths.extend(cls_imgs)
        all_labels.extend([idx] * len(cls_imgs))

    total = len(all_paths)
    print(f"  Total samples selected: {total} across {len(class_names)} classes")
    if max_samples_per_class:
        print(f"  (Sampled up to {max_samples_per_class} per class for CPU training speed)")

    # Stratified split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    print(f"  Split counts - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = _build_tf_dataset(X_train, y_train, is_train=True)
    val_ds = _build_tf_dataset(X_val, y_val, is_train=False)
    test_ds = _build_tf_dataset(X_test, y_test, is_train=False)

    # Persist class names for the API
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    names_path = SAVED_MODELS_DIR / "class_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Dataset loaded - {len(class_names)} classes")
    print(f"  Training batches  : {len(train_ds)}")
    print(f"  Validation batches: {len(val_ds)}")
    print(f"  Test batches      : {len(test_ds)}")
    print(f"{'='*60}\n")

    return train_ds, val_ds, test_ds, class_names, len(class_names)


if __name__ == "__main__":
    get_dataset()
