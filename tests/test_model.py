"""
Tests for the prediction logic in src/inference/predictor.py.

IMPORTANT: the trained model (~440MB) is NOT committed to Git (see
.gitignore) -- it's a regenerable artifact, not source code. This means it
will not exist in CI (GitHub Actions checks out the repo fresh, with no
model). Tests that need real predictions are marked skipif the model
directory is missing, so they run for real on a machine that has trained
the model (like this one), and are skipped (not failed) in CI. This is a
deliberate, honest trade-off: CI verifies what it realistically can.
"""

from pathlib import Path

import pytest

from src.inference.predictor import ClausePredictor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "legal-clause-classifier"
MODEL_AVAILABLE = MODEL_DIR.exists()


def test_predictor_raises_clear_error_if_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClausePredictor(model_dir=tmp_path / "does-not-exist")


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained model not present (expected in CI)")
def test_predictor_returns_valid_prediction_structure():
    predictor = ClausePredictor()
    result = predictor.predict(
        "This Agreement shall be governed by the laws of the State of Delaware."
    )

    assert result["category"] in predictor.id_to_label.values()
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["all_scores"]) == len(predictor.id_to_label)
    assert abs(sum(result["all_scores"].values()) - 1.0) < 0.01  # softmax probabilities sum to ~1


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained model not present (expected in CI)")
def test_predictor_batch_matches_input_count():
    predictor = ClausePredictor()
    texts = [
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "Either party may terminate this Agreement upon 30 days written notice.",
    ]
    results = predictor.predict_batch(texts)
    assert len(results) == len(texts)
    for result in results:
        assert "category" in result
        assert "confidence" in result
