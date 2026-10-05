# 🌿 Plant Disease Detection — Capstone AI System

AI-powered plant disease detection using Deep Learning. Upload or drag-and-drop any leaf image to get instant diagnosis, confidence scores, and agronomic treatment/prevention recommendations across three neural network architectures.

---

## 🚀 Quick Launch (One-Click)

On Windows, simply double-click:
```bash
start.bat
```
This automatically starts:
1. **FastAPI Backend**: `http://127.0.0.1:8000` (API docs at `/docs`)
2. **React Frontend**: `http://localhost:5173`
3. Opens your default browser automatically!

---

## 🌟 Key Features

- **3 Deep Learning Architectures**:
  - **MobileNetV2** (Transfer Learning) — **95.2% Accuracy** (18.3 MB, recommended for production/mobile).
  - **ResNet50** (Transfer Learning) — **84.2% Accuracy** (96.7 MB, deep residual architecture).
  - **Custom CNN** (Trained from Scratch) — 4-block baseline convolutional model (5.4 MB).
- **Interactive Web App**: Modern responsive dark-mode UI with glassmorphism, drag-and-drop image upload, confidence meter, and top-5 probability breakdown.
- **Agronomic Treatment & Prevention**: When a disease is detected, actionable recommendations (Cause, Organic Treatment, Chemical Control, and Long-Term Prevention) are provided.
- **Model Benchmarks & Comparison**: Built-in comparison module evaluating accuracy, precision, recall, F1 score, model size, and parameters.
- **RESTful API**: Clean FastAPI backend with async endpoints, OpenAPI Swagger documentation, and automated model loading.

---

## 📊 Model Comparison & Benchmarks

All models were evaluated on the 38-class PlantVillage dataset:

| Model | Accuracy | Precision | Recall | F1-Score | Model Size | Trainable / Total Params | Status |
|---|---|---|---|---|---|---|---|
| **MobileNetV2** ⭐ | **95.2%** | **95.1%** | **94.9%** | **95.0%** | **18.3 MB** | ~2.7 Million | **Trained & Ready** |
| **ResNet50** | **84.2%** | **83.8%** | **83.5%** | **83.6%** | **96.7 MB** | ~24.1 Million | **Trained & Ready** |
| **Custom CNN** | **7.0%** | **2.7%** | **7.0%** | **2.7%** | **5.4 MB** | ~464 Thousand | **Trained & Ready** |

*(MobileNetV2 delivers the optimal balance of ultra-high accuracy, low latency, and compact memory footprint).*

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras, Pillow, Scikit-learn
- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **Frontend**: React.js (Vite), Modern Vanilla CSS, Google Fonts
- **Dataset**: PlantVillage (38 crop condition classes)

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | `POST` | Upload plant image with `model` query param (`mobilenetv2`, `resnet50`, `custom_cnn`) → Returns disease diagnosis, confidence, top 5, and remedies |
| `/api/models` | `GET` | List all available models, sizes, and validation accuracies |
| `/api/comparison` | `GET` | Return comparative metrics across all 3 models |
| `/api/classes` | `GET` | List all 38 supported plant and disease classes |
| `/api/health` | `GET` | API health check and loaded models list |

---

## 💻 Manual Setup

### 1. Backend
```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).
