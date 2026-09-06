"""
Stage 3: Tokenization demo.

Purpose: SEE what tokenization actually does to real text, before we build
the full training script. Not used by the training pipeline itself.
"""

import pandas as pd
from transformers import AutoTokenizer

from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def main() -> None:
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    example = train_df.iloc[0]

    text = example["clause_text"]
    category = example["category"]

    print(f"\nOriginal clause text (category: {category}):")
    print(f'  "{text}"')

    tokens = tokenizer.tokenize(text)
    print(f"\nTokenized into {len(tokens)} sub-word tokens:")
    print(f"  {tokens}")

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
    )

    print(f"\ninput_ids (first 20 of {len(encoded['input_ids'])}):")
    print(f"  {encoded['input_ids'][:20]} ...")

    print(f"\nattention_mask (first 20 of {len(encoded['attention_mask'])}):")
    print(f"  {encoded['attention_mask'][:20]} ...")

    real_tokens = sum(encoded["attention_mask"])
    print(f"\nReal content tokens: {real_tokens} | Padding tokens: {len(encoded['attention_mask']) - real_tokens}")

    print("\nDecoding input_ids back to text (proves it's reversible):")
    print(f"  {tokenizer.decode(encoded['input_ids'][:real_tokens])}")


if __name__ == "__main__":
    main()
