"""
Loads the fine-tuned model (in ONNX format) once and exposes predict functions.

Kept separate from the API layer so the model-loading and inference logic
can be reused (tests, a CLI script, etc.) without depending on FastAPI.

WHY ONNX INSTEAD OF PLAIN PYTORCH: importing the full PyTorch + transformers
stack uses ~650MB of RAM before even answering one request -- too much for
a free deployment tier capped at 512MB. ONNX Runtime is a purpose-built,
much leaner inference engine (no training/autograd machinery, no PyTorch
runtime needed at all here), which brings memory use well under that
limit. See src/training/export_onnx.py for the one-time conversion step.
We still use the transformers library for TOKENIZATION only (no torch
import required for that) -- see requirements-api.txt.

Two ways to load the model:
  1. LOCAL FOLDER (default) -- used for local development
     (models/legal-clause-classifier-onnx/, produced by export_onnx.py).
  2. HUGGING FACE HUB -- used in deployment, via the MODEL_HF_REPO_ID
     environment variable (e.g. "thanmaiii/legal-clause-classifier-onnx").
"""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier-onnx"
MAX_LENGTH = 128


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ClausePredictor:
    """Loads the ONNX model/tokenizer/label map once and serves predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        hf_repo_id = os.getenv("MODEL_HF_REPO_ID")

        if hf_repo_id:
            logger.info("Loading ONNX model and tokenizer from Hugging Face Hub repo: %s", hf_repo_id)
            source_dir = Path(snapshot_download(repo_id=hf_repo_id))
        else:
            if not (model_dir / "model.onnx").exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {model_dir}, and MODEL_HF_REPO_ID is not set. "
                    "Run src/training/export_onnx.py first, or set MODEL_HF_REPO_ID."
                )
            logger.info("Loading ONNX model and tokenizer from local folder: %s", model_dir)
            source_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
        self.session = ort.InferenceSession(
            str(source_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._onnx_input_names = {i.name for i in self.session.get_inputs()}

        with open(source_dir / "label_map.json", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id_to_label = {v: k for k, v in label_map.items()}
        logger.info("Model loaded. %d categories available.", len(self.id_to_label))

    def _run_inference(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="np",
        )
        onnx_inputs = {
            name: value.astype(np.int64)
            for name, value in encoded.items()
            if name in self._onnx_input_names
        }
        logits = self.session.run(None, onnx_inputs)[0]
        return _softmax(logits)

    def predict(self, text: str) -> dict:
        """Predict the clause category for a single piece of text.

        Returns a dict with the predicted category, its confidence score,
        and the full probability distribution across all categories.
        """
        probabilities = self._run_inference([text])[0]
        predicted_id = int(np.argmax(probabilities))

        return {
            "category": self.id_to_label[predicted_id],
            "confidence": float(probabilities[predicted_id]),
            "all_scores": {
                self.id_to_label[i]: float(probabilities[i])
                for i in range(len(self.id_to_label))
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict categories for many texts in one batched inference call.

        Used when classifying all paragraphs of an uploaded contract --
        one batched call is far faster than calling predict() in a loop.
        """
        if not texts:
            return []

        probabilities = self._run_inference(texts)
        predicted_ids = np.argmax(probabilities, axis=-1)

        return [
            {
                "category": self.id_to_label[int(predicted_ids[i])],
                "confidence": float(probabilities[i, predicted_ids[i]]),
            }
            for i in range(len(texts))
        ]
