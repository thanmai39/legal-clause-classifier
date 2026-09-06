"""
Loads the fine-tuned model once and exposes a simple predict() function.

Kept separate from the API layer so the model-loading and inference logic
can be reused (tests, a CLI script, etc.) without depending on FastAPI.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier"
MAX_LENGTH = 128


class ClausePredictor:
    """Loads the model/tokenizer/label map once and serves predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        # Check for the actual weights file, not just the directory: the
        # directory can exist (e.g. containing only evaluation_report.txt /
        # confusion_matrix.png, which ARE committed to Git) without the
        # ~440MB model.safetensors file actually being present, e.g. on a
        # fresh clone or in CI, which never trains a model.
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Trained model weights not found at {model_dir}. "
                "Run src/training/train.py first to produce a trained model."
            )

        logger.info("Loading model and tokenizer from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

        with open(model_dir / "label_map.json", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id_to_label = {v: k for k, v in label_map.items()}
        logger.info("Model loaded. %d categories available.", len(self.id_to_label))

    def predict(self, text: str) -> dict:
        """Predict the clause category for a single piece of text.

        Returns a dict with the predicted category, its confidence score,
        and the full probability distribution across all categories.
        """
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = F.softmax(logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        predicted_category = self.id_to_label[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        all_scores = {
            self.id_to_label[i]: float(probabilities[i].item())
            for i in range(len(self.id_to_label))
        }

        return {
            "category": predicted_category,
            "confidence": confidence,
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict categories for many texts in one forward pass.

        Used when classifying all paragraphs of an uploaded contract --
        one batched forward pass is far faster than calling predict()
        in a loop, since the model processes the whole batch together.
        """
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = F.softmax(logits, dim=-1)

        predicted_ids = torch.argmax(probabilities, dim=-1)

        results = []
        for i in range(len(texts)):
            predicted_id = int(predicted_ids[i].item())
            results.append({
                "category": self.id_to_label[predicted_id],
                "confidence": float(probabilities[i, predicted_id].item()),
            })
        return results
