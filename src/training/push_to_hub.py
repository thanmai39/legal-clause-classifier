"""
One-time script: push the trained model to Hugging Face Hub.

WHY: our trained model (~440MB) is intentionally NOT committed to Git --
GitHub has a hard 100MB per-file limit, and large binary model weights
don't belong in source control anyway. For deployment, a cloud platform
building our Docker image from GitHub has no access to this local file.
Hugging Face Hub (the same place we downloaded LegalBERT and CUAD from)
is a free, standard place to host trained model artifacts, and our API
can download from there at startup instead of expecting a local folder.

Usage:
    python src/training/push_to_hub.py <your-hf-username>/legal-clause-classifier

Requires HF_TOKEN to be set in .env (a Hugging Face access token with
WRITE permission -- see https://huggingface.co/settings/tokens).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/training/push_to_hub.py <username>/<repo-name>")
        sys.exit(1)

    repo_id = sys.argv[1]

    if not (MODEL_DIR / "model.safetensors").exists():
        print(f"No trained model found at {MODEL_DIR}. Run src/training/train.py first.")
        sys.exit(1)

    api = HfApi()
    print(f"Creating (or reusing) repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {MODEL_DIR} -> {repo_id} (this uploads ~440MB, may take a few minutes)...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        # Don't upload our own evaluation artifacts -- they're for the repo's
        # docs, not needed for the model to load and predict.
        ignore_patterns=["evaluation_report.txt", "confusion_matrix.png"],
    )

    print(f"\nDone. Model available at: https://huggingface.co/{repo_id}")
    print(f"Set MODEL_HF_REPO_ID={repo_id} in your deployment environment to use it.")


if __name__ == "__main__":
    main()
