"""
Tests for the FastAPI endpoints in src/api/main.py.

Split deliberately into two groups:
  - Tests that work regardless of whether a trained model exists (health
    check, input validation, error handling) -- these run everywhere,
    including CI.
  - Tests that need real predictions -- skipped if the model isn't present
    (see the MODEL_AVAILABLE note in test_model.py for why).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "legal-clause-classifier"
MODEL_AVAILABLE = MODEL_DIR.exists()


def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "explainer_available" in body


def test_predict_rejects_empty_text():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # fails Pydantic's min_length validation


def test_predict_rejects_missing_text_field():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422


def test_predict_batch_rejects_empty_list():
    with TestClient(app) as client:
        response = client.post("/predict-batch", json={"texts": []})
        assert response.status_code == 422


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained model not present (expected in CI)")
def test_predict_returns_valid_response_structure():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "This Agreement shall be governed by the laws of the State of Delaware."},
        )
        assert response.status_code == 200
        body = response.json()
        assert "category" in body
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert "all_scores" in body


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained model not present (expected in CI)")
def test_predict_batch_returns_one_result_per_input():
    with TestClient(app) as client:
        texts = [
            "Governing Law shall be the State of Delaware.",
            "Either party may terminate with 30 days notice.",
        ]
        response = client.post("/predict-batch", json={"texts": texts})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == len(texts)
