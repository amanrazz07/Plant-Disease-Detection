"""
FastAPI entry point for the Plant Disease Detection API.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.predict import router as predict_router
from app.models.inference import model_manager


# ---------------------------------------------------------------------------
# Lifespan — pre-load models on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log available models on startup without blocking server bind."""
    print(f"\n[Plant App] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"   Models directory: {settings.MODELS_DIR}")
    available = [m for m in settings.AVAILABLE_MODELS if model_manager.is_available(m)]
    if available:
        print(f"   [OK] Available trained models: {', '.join(available)}")
    else:
        print("   [!] No trained models found -- train models first!")
    print()
    yield
    print("[Plant App] Shutting down...")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
)

# CORS - allow all origins so deployed Vercel frontend and localhost work seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predict_router, prefix=settings.API_PREFIX, tags=["Prediction"])


# Root redirect
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": f"🌿 {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
