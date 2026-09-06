"""
Stage 2: Turn raw CUAD data into a clean, split classification dataset.

Implementation decision for THIS project (not part of original CUAD):
CUAD was built for extractive question-answering ("find the span in this
contract related to category X"). We reshape it into a single-label
classification dataset of (clause_text -> category) pairs by taking each
non-empty extracted answer as one example, labeled with the category asked
about in its question.

Pipeline:
  raw jsonl (train + test, as shipped by CUAD)
    -> combine into one pool
    -> extract (clause_text, category) from non-empty answers
    -> clean whitespace
    -> drop near-empty/junk answers (len <= 1 char)
    -> drop categories with too few examples (< MIN_EXAMPLES_PER_CATEGORY)
       Why: a category with only a handful of examples cannot be reliably
       split into train/val/test or learned by the model. This is a
       documented trade-off, not silent data loss -- see the printed report.
    -> stratified split into train/val/test (70/15/15), so every category
       is proportionally represented in each split despite the imbalance
    -> save data/processed/{train,val,test}.csv and label_map.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

MIN_EXAMPLES_PER_CATEGORY = 100
MIN_ANSWER_LENGTH = 2  # drop answers shorter than this (junk, e.g. stray punctuation)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

CATEGORY_PATTERN = re.compile(r'"([^"]+)"')


def extract_category(question: str) -> str:
    match = CATEGORY_PATTERN.search(question)
    return match.group(1) if match else "UNKNOWN"


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted clause text.

    Contracts contain irregular line breaks, repeated spaces, and page
    artifacts from PDF extraction. We collapse all whitespace runs to a
    single space and strip leading/trailing whitespace. We deliberately do
    NOT lowercase or strip punctuation -- transformer tokenizers are
    trained on natural, cased text and legal punctuation (e.g. section
    numbers, "Section 4.2(a)") can be meaningful.
    """
    return re.sub(r"\s+", " ", text).strip()


def load_raw_examples() -> pd.DataFrame:
    rows = []
    for split_file in ("cuad_train.jsonl", "cuad_test.jsonl"):
        path = RAW_DIR / split_file
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                answer_texts = record["answers"].get("text", [])
                if not answer_texts:
                    continue
                raw_answer = answer_texts[0]
                if not raw_answer or not raw_answer.strip():
                    continue

                cleaned = clean_text(raw_answer)
                if len(cleaned) < MIN_ANSWER_LENGTH:
                    continue

                rows.append({
                    "id": record["id"],
                    "category": extract_category(record["question"]),
                    "clause_text": cleaned,
                })

    return pd.DataFrame(rows)


def filter_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(df["category"])
    kept_categories = {c for c, n in counts.items() if n >= MIN_EXAMPLES_PER_CATEGORY}
    dropped_categories = {c: n for c, n in counts.items() if n < MIN_EXAMPLES_PER_CATEGORY}

    print(f"\nCategories kept ({len(kept_categories)}):")
    for c in sorted(kept_categories):
        print(f"  {counts[c]:5d}  {c}")

    print(f"\nCategories DROPPED for having < {MIN_EXAMPLES_PER_CATEGORY} examples ({len(dropped_categories)}):")
    for c, n in sorted(dropped_categories.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {c}")

    return df[df["category"].isin(kept_categories)].reset_index(drop=True)


def stratified_split(df: pd.DataFrame):
    # First split off the training set, then split the remainder into val/test.
    train_df, remaining_df = train_test_split(
        df,
        train_size=TRAIN_FRAC,
        stratify=df["category"],
        random_state=42,
    )
    relative_val_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=relative_val_frac,
        stratify=remaining_df["category"],
        random_state=42,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning raw examples...")
    df = load_raw_examples()
    print(f"Total usable (non-empty, non-junk) examples: {len(df)}")

    df = filter_rare_categories(df)
    print(f"\nExamples remaining after category filtering: {len(df)}")

    train_df, val_df, test_df = stratified_split(df)

    categories = sorted(df["category"].unique())
    label_map = {category: idx for idx, category in enumerate(categories)}

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    with open(PROCESSED_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nFinal split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Number of classes: {len(label_map)}")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
