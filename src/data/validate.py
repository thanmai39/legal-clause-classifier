"""
Stage 2: Validate the raw CUAD data.

This script only INSPECTS and REPORTS on data/raw/ — it does not modify or
save anything. The goal is to understand the shape and quality of the data
before we write any cleaning/transformation logic in preprocess.py.

CUAD structure reminder: each row asks "does contract X contain clause type Y?"
- question  -> which clause category is being asked about
- context   -> the full contract text
- answers   -> the extracted clause text, EMPTY if that contract has no such clause

Implementation decision (this project, not part of original CUAD):
We only keep rows with a non-empty answer, since we're building a
(clause_text -> category) CLASSIFICATION dataset, not the original
extractive-QA task CUAD was designed for.
"""

import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CATEGORY_PATTERN = re.compile(r'"([^"]+)"')


def extract_category(question: str) -> str:
    """Pull the clause category name out of a CUAD question string.

    Example question:
      'Highlight the parts (if any) of this contract related to
       "Governing Law" that should be reviewed by a lawyer. Details: ...'
    -> "Governing Law"
    """
    match = CATEGORY_PATTERN.search(question)
    return match.group(1) if match else "UNKNOWN"


def validate_split(path: Path) -> None:
    print(f"\n=== Validating {path.name} ===")

    total_rows = 0
    empty_answers = 0
    non_empty_answers = 0
    missing_fields = 0
    category_counts = Counter()
    context_lengths = []
    answer_lengths = []
    ids_seen = set()
    duplicate_ids = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            row = json.loads(line)

            required = ("id", "question", "context", "answers")
            if not all(k in row for k in required):
                missing_fields += 1
                continue

            if row["id"] in ids_seen:
                duplicate_ids += 1
            ids_seen.add(row["id"])

            answer_texts = row["answers"].get("text", [])
            if not answer_texts or not answer_texts[0].strip():
                empty_answers += 1
                continue

            non_empty_answers += 1
            category = extract_category(row["question"])
            category_counts[category] += 1
            context_lengths.append(len(row["context"]))
            answer_lengths.append(len(answer_texts[0]))

    print(f"Total rows:               {total_rows}")
    print(f"Rows with missing fields: {missing_fields}")
    print(f"Duplicate ids:            {duplicate_ids}")
    print(f"Rows with EMPTY answer:   {empty_answers}  (clause type absent in that contract)")
    print(f"Rows with an answer:      {non_empty_answers}  (usable for classification)")
    print(f"Number of distinct clause categories found: {len(category_counts)}")

    if answer_lengths:
        print(f"Answer length (chars):  min={min(answer_lengths)}, "
              f"max={max(answer_lengths)}, avg={sum(answer_lengths)/len(answer_lengths):.0f}")
    if context_lengths:
        print(f"Context length (chars): min={min(context_lengths)}, "
              f"max={max(context_lengths)}, avg={sum(context_lengths)/len(context_lengths):.0f}")

    print("\nTop 10 most common categories (usable rows):")
    for category, count in category_counts.most_common(10):
        print(f"  {count:5d}  {category}")

    print("\nBottom 10 least common categories (usable rows) -- class imbalance preview:")
    for category, count in category_counts.most_common()[-10:]:
        print(f"  {count:5d}  {category}")


if __name__ == "__main__":
    for split_file in ("cuad_train.jsonl", "cuad_test.jsonl"):
        validate_split(RAW_DIR / split_file)
