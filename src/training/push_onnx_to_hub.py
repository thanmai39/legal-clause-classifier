"""
One-time script: push the quantized ONNX model to Hugging Face Hub for
deployment.

Usage:
    python src/training/push_onnx_to_hub.py <your-hf-username>/legal-clause-classifier-onnx

Requires HF_TOKEN in .env (write-permission token).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

ONNX_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier-onnx"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/training/push_onnx_to_hub.py <username>/<repo-name>")
        sys.exit(1)

    repo_id = sys.argv[1]

    if not (ONNX_DIR / "model.onnx").exists():
        print(f"No ONNX model found at {ONNX_DIR}. Run src/training/export_onnx.py first.")
        sys.exit(1)

    api = HfApi()
    print(f"Creating (or reusing) repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {ONNX_DIR} -> {repo_id}...")
    api.upload_folder(folder_path=str(ONNX_DIR), repo_id=repo_id, repo_type="model")

    print(f"\nDone. Model available at: https://huggingface.co/{repo_id}")
    print(f"Set MODEL_HF_REPO_ID={repo_id} in your deployment environment.")


if __name__ == "__main__":
    main()
