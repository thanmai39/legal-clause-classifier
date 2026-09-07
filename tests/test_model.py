"""
Tests for the prediction logic in src/inference/predictor.py.

IMPORTANT: the trained model weights (~440MB) are NOT committed to Git
(see .gitignore) -- they're a regenerable artifact, not source code. Only
evaluation_report.txt and confusion_matrix.png are committed from that
folder, so the DIRECTORY itself exists even in a fresh checkout with no
model. We therefore check for the actual weights file (model.safetensors),
not just the directory, to decide whether real predictions can run. Tests
that need real predictions are skipped (not failed) when it's absent --
e.g. in CI, which never trains a model. This is a deliberate, honest
trade-off: CI verifies what it realistically can.
"""

from pathlib import Path

import pytest

from src.inference.predictor import ClausePredictor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "legal-clause-classifier"
MODEL_AVAILABLE = (MODEL_DIR / "model.safetensors").exists()


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
