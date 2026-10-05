# 🌿 Plant Disease Detection — Backend

AI-powered plant disease detection using deep learning. This backend serves three trained models (Custom CNN, MobileNetV2, ResNet50) via a FastAPI REST API.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Models

Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract it to `data/PlantVillage/`.

```bash
# Train all three models
python -m ml.train_cnn --source directory --data-dir data/PlantVillage --epochs 50
python -m ml.train_mobilenet --source directory --data-dir data/PlantVillage --epochs 30
python -m ml.train_resnet --source directory --data-dir data/PlantVillage --epochs 30

# Generate comparison report
python -m ml.compare_models
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Upload image → disease prediction + confidence |
| `/api/models` | GET | List available models |
| `/api/health` | GET | API health check |
| `/api/classes` | GET | Get all 38 disease class names |

## Project Structure

```
backend/
├── app/                    # FastAPI application
│   ├── main.py             # Entry point
│   ├── api/predict.py      # API routes
│   ├── models/inference.py # Model loading & prediction
│   ├── schemas/            # Pydantic schemas
│   └── core/config.py      # Settings
├── ml/                     # Training pipeline
│   ├── preprocess.py       # Data loading & augmentation
│   ├── train_cnn.py        # Custom CNN
│   ├── train_mobilenet.py  # MobileNetV2
│   ├── train_resnet.py     # ResNet50
│   ├── compare_models.py   # Comparison report
│   └── utils.py            # Shared utilities
├── saved_models/           # Trained .keras files
└── results/                # Metrics, plots, CSVs
```
