"""
FastAPI backend for the Legal Clause Classifier.

Endpoints:
  GET  /health   -> liveness/readiness check
  POST /predict  -> classify a clause of text into a legal category
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import ClausePredictor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Holds the loaded model. Populated once at startup (see lifespan below),
# not on every request -- loading a transformer model takes real time and
# should not happen per-call.
_predictor: ClausePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    logger.info("Application startup: loading model...")
    try:
        _predictor = ClausePredictor()
        logger.info("Model loaded successfully. API ready to serve predictions.")
    except FileNotFoundError as e:
        # Don't crash the whole API if the model isn't trained yet --
        # /health will still work, but /predict will report 503 until
        # a model is available. This matters for local dev and for
        # container startup order.
        logger.error("Model not available at startup: %s", e)
        _predictor = None
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="Legal Clause Classifier API",
    description="Classifies legal contract clauses into categories using a fine-tuned LegalBERT model.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=5000, description="Clause text to classify")


class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_predictor is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _predictor is None:
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Has training been run yet?",
        )

    start_time = time.time()
    try:
        result = _predictor.predict(request.text)
    except Exception:
        logger.exception("Prediction failed due to an internal error.")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

    elapsed = time.time() - start_time
    logger.info(
        "Prediction completed | category=%s | confidence=%.3f | took=%.3fs",
        result["category"], result["confidence"], elapsed,
    )
    return PredictResponse(**result)
