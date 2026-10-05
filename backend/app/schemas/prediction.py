"""Pydantic schemas for prediction request / response objects."""

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single class prediction."""
    class_name: str = Field(..., description="Raw class label, e.g. 'Tomato___Early_blight'")
    plant: str = Field(..., description="Plant name, e.g. 'Tomato'")
    condition: str = Field(..., description="Disease or 'Healthy'")
    is_healthy: bool = Field(..., description="Whether the plant is healthy")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence 0-1")


class DiseaseRemedy(BaseModel):
    """Agronomic recommendations for the predicted condition."""
    cause: str
    treatment: str
    prevention: str


class PredictionResponse(BaseModel):
    """Full response returned by the /api/predict endpoint."""
    success: bool
    model_used: str
    prediction: PredictionResult
    remedy: DiseaseRemedy | None = None
    top_5: list[PredictionResult] = Field(
        default_factory=list,
        description="Top-5 predictions sorted by confidence",
    )


class ComparisonItem(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_ms: float
    size_mb: float
    parameters: str


class ComparisonResponse(BaseModel):
    models: list[ComparisonItem]


class ModelInfo(BaseModel):
    """Metadata for a single available model."""
    name: str
    display_name: str
    description: str
    available: bool
    size_mb: float | None = None
    accuracy: float | None = None


class ModelsResponse(BaseModel):
    """Response for /api/models."""
    models: list[ModelInfo]
    default_model: str


class HealthResponse(BaseModel):
    """Response for /api/health."""
    status: str
    version: str
    models_loaded: list[str]


class ClassInfo(BaseModel):
    """Single class information."""
    index: int
    raw_name: str
    plant: str
    condition: str
    is_healthy: bool


class ClassesResponse(BaseModel):
    """Response for /api/classes."""
    total: int
    classes: list[ClassInfo]
