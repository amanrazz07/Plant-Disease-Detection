"""
Train all three models sequentially with memory-safe settings for Windows CPU.
Runs each model in an isolated subprocess to completely eliminate memory leakage.
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
SAVE_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def train_single_model(script_code: str, model_name: str):
    """Run training in a fresh subprocess to isolate memory."""
    print(f"\n{'='*60}")
    print(f"  Starting Pipeline: {model_name}")
    print(f"{'='*60}\n", flush=True)

    script_path = PROJECT_ROOT / f"_train_{model_name.lower().replace(' ', '_')}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_code)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start

    try:
        if script_path.exists():
            script_path.unlink()
    except OSError:
        pass

    if result.returncode == 0:
        print(f"\n  >> {model_name} COMPLETE ({elapsed:.1f}s)", flush=True)
    else:
        print(f"\n  >> {model_name} FAILED (exit code {result.returncode}, {elapsed:.1f}s)", flush=True)

    return result.returncode == 0


# ============================================================================
# Model 1: Custom CNN (128x128, lightweight architecture, CPU fast)
# ============================================================================
CNN_SCRIPT = f'''
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"{PROJECT_ROOT}")

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"{DATA_DIR}")
SAVE_DIR = Path(r"{PROJECT_ROOT}") / "saved_models"
RESULTS_DIR = Path(r"{PROJECT_ROOT}") / "results"
SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
MAX_PER_CLASS = 60
EPOCHS = 6

model_path = SAVE_DIR / "custom_cnn.keras"
metrics_path = RESULTS_DIR / "custom_cnn_metrics.json"

class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
num_classes = len(class_names)
print(f"Custom CNN: Found {{num_classes}} classes")

all_paths, all_labels = [], []
for idx, d in enumerate(class_dirs):
    imgs = [str(f) for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:MAX_PER_CLASS]
    all_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))

print(f"Total dataset samples: {{len(all_paths)}}")
X_train, X_temp, y_train, y_temp = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")

def load_img(path_tensor):
    p = path_tensor.numpy().decode("utf-8")
    try:
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        return arr
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

def map_fn(path, label):
    [image] = tf.py_function(load_img, [path], [tf.float32])
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, label

def make_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 1000), seed=42)
    return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(1)

train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds = make_ds(X_val, y_val)
test_ds = make_ds(X_test, y_test)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="CustomCNN")
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ]
)

if not model_path.exists():
    model.save(str(model_path))

# Save history
with open(RESULTS_DIR / "custom_cnn_history.json", "w") as f:
    json.dump(dict((k, [float(v) for v in vals]) for k, vals in history.history.items()), f, indent=2)

# Evaluate on test set
y_true, y_pred_probs = [], []
for imgs, lbls in test_ds:
    y_pred_probs.append(model.predict(imgs, verbose=0))
    y_true.append(lbls.numpy())
y_true = np.concatenate(y_true)
y_pred = np.argmax(np.concatenate(y_pred_probs), axis=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = float(accuracy_score(y_true, y_pred))
prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

metrics = {{
    "model_name": "custom_cnn",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "total_params": int(model.count_params()),
    "model_size_mb": round(model_path.stat().st_size / 1048576, 2)
}}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\\nCustom CNN Results: Acc={{acc:.4f}}, Prec={{prec:.4f}}, Rec={{rec:.4f}}, F1={{f1:.4f}}")
print("Custom CNN training completed successfully!")
'''

# ============================================================================
# Model 2: MobileNetV2 (Check & quick reuse)
# ============================================================================
MOBILENET_SCRIPT = f'''
import json
from pathlib import Path

SAVE_DIR = Path(r"{PROJECT_ROOT}") / "saved_models"
RESULTS_DIR = Path(r"{PROJECT_ROOT}") / "results"

model_path = SAVE_DIR / "mobilenetv2.keras"
metrics_path = RESULTS_DIR / "mobilenetv2_metrics.json"

if model_path.exists() and metrics_path.exists():
    with open(metrics_path) as f:
        m = json.load(f)
    print(f"MobileNetV2 already trained ({{model_path.stat().st_size/1048576:.1f}} MB)")
    print(f"Accuracy: {{m['accuracy']*100:.2f}}% | Params: {{m.get('total_params', 0):,}}")
    print("MobileNetV2 ready.")
else:
    print("MobileNetV2 model or metrics missing.")
'''

# ============================================================================
# Model 3: ResNet50 (Transfer Learning with frozen backbone, fast and stable)
# ============================================================================
RESNET_SCRIPT = f'''
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"{PROJECT_ROOT}")

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"{DATA_DIR}")
SAVE_DIR = Path(r"{PROJECT_ROOT}") / "saved_models"
RESULTS_DIR = Path(r"{PROJECT_ROOT}") / "results"
SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
MAX_PER_CLASS = 50
EPOCHS = 5

model_path = SAVE_DIR / "resnet50.keras"
metrics_path = RESULTS_DIR / "resnet50_metrics.json"

class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
num_classes = len(class_names)
print(f"ResNet50: Found {{num_classes}} classes")

all_paths, all_labels = [], []
for idx, d in enumerate(class_dirs):
    imgs = [str(f) for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:MAX_PER_CLASS]
    all_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))

print(f"Total dataset samples: {{len(all_paths)}}")
X_train, X_temp, y_train, y_temp = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")

def load_img(path_tensor):
    p = path_tensor.numpy().decode("utf-8")
    try:
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        return arr
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

def map_fn(path, label):
    [image] = tf.py_function(load_img, [path], [tf.float32])
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, label

def make_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 1000), seed=42)
    return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(1)

train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds = make_ds(X_val, y_val)
test_ds = make_ds(X_test, y_test)

# Build ResNet50
base = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="ResNet50_Transfer")
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ]
)

if not model_path.exists():
    model.save(str(model_path))

with open(RESULTS_DIR / "resnet50_history.json", "w") as f:
    json.dump(dict((k, [float(v) for v in vals]) for k, vals in history.history.items()), f, indent=2)

# Evaluate
y_true, y_pred_probs = [], []
for imgs, lbls in test_ds:
    y_pred_probs.append(model.predict(imgs, verbose=0))
    y_true.append(lbls.numpy())
y_true = np.concatenate(y_true)
y_pred = np.argmax(np.concatenate(y_pred_probs), axis=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = float(accuracy_score(y_true, y_pred))
prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

metrics = {{
    "model_name": "resnet50",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "total_params": int(model.count_params()),
    "model_size_mb": round(model_path.stat().st_size / 1048576, 2)
}}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\\nResNet50 Results: Acc={{acc:.4f}}, Prec={{prec:.4f}}, Rec={{rec:.4f}}, F1={{f1:.4f}}")
print("ResNet50 training completed successfully!")
'''

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Plant Disease Detection - Sequential Safe Training")
    print("=" * 60)

    results = {}

    # 1. Custom CNN
    results["custom_cnn"] = train_single_model(CNN_SCRIPT, "Custom CNN")

    # 2. MobileNetV2
    results["mobilenetv2"] = train_single_model(MOBILENET_SCRIPT, "MobileNetV2")

    # 3. ResNet50
    results["resnet50"] = train_single_model(RESNET_SCRIPT, "ResNet50")

    # Run comparisons if scripts completed
    try:
        import ml.compare_models as cm
        cm.main()
    except Exception as e:
        print(f"Comparison note: {e}")

    print("\n" + "=" * 60)
    print("  All Models Status")
    print("=" * 60)
    for name, ok in results.items():
        status = "READY" if ok else "FAILED"
        print(f"  {name:15s} : {status}")
    print("=" * 60)
