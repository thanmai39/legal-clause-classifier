"""
Stage 2: Download the raw CUAD dataset and save it, untouched, into data/raw/.

This script ONLY downloads and saves the dataset in its original form.
No cleaning, filtering, or transformation happens here — that belongs in
preprocess.py. Keeping "get the data" and "change the data" as separate
steps is what makes data/raw/ trustworthy: it should always be possible to
delete data/processed/ and regenerate it from data/raw/ without re-downloading.
"""

from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_cuad() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CUAD dataset from Hugging Face (theatticusproject/cuad-qa)...")
    dataset = load_dataset("theatticusproject/cuad-qa", revision="refs/convert/parquet")

    print("Dataset splits found:", list(dataset.keys()))

    for split_name, split_data in dataset.items():
        out_path = RAW_DIR / f"cuad_{split_name}.jsonl"
        split_data.to_json(str(out_path))
        print(f"Saved split '{split_name}' ({len(split_data)} rows) -> {out_path}")


if __name__ == "__main__":
    download_cuad()
