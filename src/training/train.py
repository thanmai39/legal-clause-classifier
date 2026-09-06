"""
Stage 3: Fine-tune LegalBERT on the processed CUAD clause classification data.

Pipeline: clause_text -> tokenize -> LegalBERT -> classification head -> category

Class imbalance handling (implementation decision for this project):
We use a WEIGHTED cross-entropy loss during training instead of oversampling
the raw text. Rare categories get a higher weight, so the model is penalized
more for getting them wrong -- without duplicating any text examples, which
for transformer fine-tuning tends to cause overfitting on the duplicated
examples.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "models" / "checkpoints"

BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8


class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedLossTrainer(Trainer):
    """A Trainer that applies class weights to the loss function.

    Standard Trainer uses plain (unweighted) cross-entropy, which treats
    every example equally regardless of class frequency. We override
    compute_loss to weight rare classes more heavily.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def main() -> None:
    with open(PROCESSED_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = len(label_map)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    train_df["label_id"] = train_df["category"].map(label_map)
    val_df["label_id"] = val_df["category"].map(label_map)

    print(f"Loading tokenizer and base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=num_labels
    )

    print("Tokenizing training and validation sets...")
    train_encodings = tokenizer(
        list(train_df["clause_text"]),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    val_encodings = tokenizer(
        list(val_df["clause_text"]),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    train_dataset = ClauseDataset(train_encodings, list(train_df["label_id"]))
    val_dataset = ClauseDataset(val_encodings, list(val_df["label_id"]))

    present_classes = np.unique(train_df["label_id"].values)
    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=train_df["label_id"].values,
    )
    # Default weight 1.0 for any label_id absent from the training set
    # (defensive; with our thresholded categories this shouldn't happen,
    # but a silent index mismatch here would corrupt every loss value).
    class_weights = np.ones(num_labels, dtype=np.float32)
    for cls_id, weight in zip(present_classes, present_weights):
        class_weights[cls_id] = weight
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"\nComputed class weights (min={class_weights.min():.2f}, max={class_weights.max():.2f})")
    print("Rare categories get higher weight -> penalized more when misclassified.")

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=[],
        save_total_limit=2,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nTraining complete. Final validation metrics:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    with open(MODEL_OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved final model, tokenizer, and label_map.json to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
