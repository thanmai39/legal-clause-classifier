"""
Stage 3: Evaluate the fine-tuned model on the held-out TEST set.

This is data the model has never seen during training OR validation
(validation was used during training to pick the best checkpoint; test is
used exactly once, here, to get an honest final measurement).

Produces:
  - overall accuracy, macro-F1, weighted-F1
  - per-class precision/recall/F1 (sklearn classification_report)
  - confusion matrix (saved as PNG)
All results are also saved to models/legal-clause-classifier/evaluation_report.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier"

MAX_LENGTH = 128
BATCH_SIZE = 16


def main() -> None:
    with open(MODEL_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}

    print(f"Loading model and tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_df["label_id"] = test_df["category"].map(label_map)
    true_labels = test_df["label_id"].values

    print(f"Running inference on {len(test_df)} test examples...")
    predictions = []
    texts = list(test_df["clause_text"])

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            batch_preds = torch.argmax(logits, dim=-1).numpy()
            predictions.extend(batch_preds)

    predictions = np.array(predictions)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    weighted_f1 = f1_score(true_labels, predictions, average="weighted")

    label_names = [id_to_label[i] for i in sorted(id_to_label)]
    report = classification_report(
        true_labels, predictions, target_names=label_names, digits=3, zero_division=0
    )

    summary_lines = [
        "=== Test Set Evaluation ===",
        f"Accuracy:     {accuracy:.4f}",
        f"Macro-F1:     {macro_f1:.4f}",
        f"Weighted-F1:  {weighted_f1:.4f}",
        "",
        "Per-class precision/recall/F1:",
        report,
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    report_path = MODEL_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nSaved full report to {report_path}")

    cm = confusion_matrix(true_labels, predictions, labels=sorted(id_to_label))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion Matrix - Test Set")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
