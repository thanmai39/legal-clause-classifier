"""
FastAPI backend for the Legal Clause Classifier.

Endpoints:
  GET  /health         -> liveness/readiness check
  POST /predict        -> classify a single clause of text into a legal category
  POST /predict-batch  -> classify many clauses at once (e.g. paragraphs of an uploaded contract)
  POST /explain        -> plain-English explanation of an already-predicted category (via Gemini)
"""

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()  # reads .env into environment variables before anything reads them

from src.inference.explainer import GeminiExplainer
from src.inference.predictor import ClausePredictor
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Holds the loaded model/explainer. Populated once at startup (see lifespan
# below), not on every request -- loading a transformer model takes real
# time and should not happen per-call.
_predictor: ClausePredictor | None = None
_explainer: GeminiExplainer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _explainer
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

    _explainer = GeminiExplainer()  # logs its own warning if no API key is set
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


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=200,
        description="List of clause/paragraph texts to classify (max 200 per request)",
    )


class BatchPredictionItem(BaseModel):
    category: str
    confidence: float


class PredictBatchResponse(BaseModel):
    results: list[BatchPredictionItem]


class ExplainRequest(BaseModel):
    category: str = Field(..., description="Predicted category (from /predict)")
    text: str = Field(..., min_length=2, max_length=5000, description="The clause text")


class ExplainResponse(BaseModel):
    explanation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_available: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        explainer_available=_explainer is not None and _explainer.is_available(),
    )


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


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if _predictor is None:
        logger.error("Predict-batch called but model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Has training been run yet?",
        )

    start_time = time.time()
    try:
        results = _predictor.predict_batch(request.texts)
    except Exception:
        logger.exception("Batch prediction failed due to an internal error.")
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error.")

    elapsed = time.time() - start_time
    logger.info(
        "Batch prediction completed | count=%d | took=%.3fs",
        len(request.texts), elapsed,
    )
    return PredictBatchResponse(results=[BatchPredictionItem(**r) for r in results])


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    if _explainer is None or not _explainer.is_available():
        logger.error("Explain called but Gemini is not configured.")
        raise HTTPException(
            status_code=503,
            detail="Explanation feature is not available (GEMINI_API_KEY not configured).",
        )

    start_time = time.time()
    try:
        explanation_text = _explainer.explain(request.category, request.text)
    except Exception:
        logger.exception("Explanation generation failed due to an internal error.")
        raise HTTPException(status_code=500, detail="Explanation generation failed.")

    elapsed = time.time() - start_time
    logger.info("Explanation generated | category=%s | took=%.3fs", request.category, elapsed)
    return ExplainResponse(explanation=explanation_text)
