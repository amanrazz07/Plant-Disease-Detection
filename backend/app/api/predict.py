"""
Prediction API endpoints.

Routes:
    POST /api/predict   — upload image → disease prediction
    GET  /api/models    — list available models
    GET  /api/health    — API health check
    GET  /api/classes   — list all disease class names
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from app.models.inference import model_manager, MODEL_DISPLAY_NAMES
from app.schemas.prediction import (
    PredictionResponse,
    PredictionResult,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    ClassesResponse,
    ClassInfo,
)
from app.core.config import settings

router = APIRouter()

# Allowed image MIME types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Plant leaf image"),
    model: str = Query(
        default=settings.DEFAULT_MODEL,
        description="Model to use for prediction",
    ),
):
    """Upload a plant leaf image and get disease prediction."""
    # Validate model name
    if model not in settings.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Choose from: {settings.AVAILABLE_MODELS}",
        )

    # Check model availability
    if not model_manager.is_available(model):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' has not been trained yet. "
            f"Run the corresponding training script first.",
        )

    # Validate file type
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{file.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run prediction
    try:
        result = model_manager.predict(image_bytes, model_name=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        success=True,
        model_used=result["model_used"],
        prediction=PredictionResult(**result["prediction"]),
        remedy=result.get("remedy"),
        top_5=[PredictionResult(**r) for r in result["top_5"]],
    )


# ---------------------------------------------------------------------------
# GET /api/comparison
# ---------------------------------------------------------------------------
@router.get("/comparison")
async def get_model_comparison():
    """Return performance benchmark data across all 3 deep learning models."""
    csv_path = settings.RESULTS_DIR / "comparison_summary.csv"
    if csv_path.exists():
        import csv
        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "model": r.get("Model"),
                    "accuracy": float(r.get("Accuracy", 0)),
                    "precision": float(r.get("Precision", 0)),
                    "recall": float(r.get("Recall", 0)),
                    "f1_score": float(r.get("F1 Score", 0)),
                    "inference_ms": float(r.get("Inference (ms)", 0)),
                    "size_mb": float(r.get("Size (MB)", 0)),
                    "parameters": r.get("Parameters", "N/A"),
                })
        return {"success": True, "comparison": rows}

    return {"success": False, "comparison": []}


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------
@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models and their metadata."""
    models = []
    for name in settings.AVAILABLE_MODELS:
        info = model_manager.get_model_info(name)
        models.append(ModelInfo(**info))

    return ModelsResponse(models=models, default_model=settings.DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded=model_manager.loaded_models,
    )


# ---------------------------------------------------------------------------
# GET /api/classes
# ---------------------------------------------------------------------------
@router.get("/classes", response_model=ClassesResponse)
async def list_classes():
    """List all disease classes the models can predict."""
    classes = []
    for idx, raw in enumerate(model_manager.class_names):
        parts = raw.split("___")
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
        classes.append(
            ClassInfo(
                index=idx,
                raw_name=raw,
                plant=plant,
                condition=condition.title(),
                is_healthy=condition.lower() == "healthy",
            )
        )

    return ClassesResponse(total=len(classes), classes=classes)
