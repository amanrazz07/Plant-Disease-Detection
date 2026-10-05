"""
Application configuration powered by pydantic-settings.

Settings are loaded from environment variables and / or a ``.env`` file
located in the backend root directory.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    APP_NAME: str = "Plant Disease Detection API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI-powered plant disease detection using CNN, MobileNetV2, and ResNet50"
    )
    DEBUG: bool = False

    # Paths (derived at import time so they work from any cwd)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"

    # Model defaults
    DEFAULT_MODEL: str = "mobilenetv2"
    AVAILABLE_MODELS: list[str] = ["custom_cnn", "mobilenetv2", "resnet50"]

    # Image settings
    IMG_SIZE: int = 224

    # API
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
