"""
One-time script: export the trained PyTorch model to a QUANTIZED ONNX
model for lightweight CPU-only deployment.

WHY: the full PyTorch + transformers stack uses ~650MB+ of RAM just to
import the library and load our model -- too much for a free deployment
tier capped at 512MB RAM. Two changes fix this:
  1. ONNX Runtime instead of PyTorch for inference -- a purpose-built,
     much leaner engine (no training/autograd machinery needed at all
     in the deployed container).
  2. INT8 quantization -- storing weights as 8-bit integers instead of
     32-bit floats, roughly a 4x size/memory reduction (~418MB -> ~110MB),
     with a small, usually negligible, accuracy trade-off for a
     classification task like this.

Usage:
    python src/training/export_onnx.py
"""

import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier"
ONNX_DIR = Path(__file__).resolve().parents[2] / "models" / "legal-clause-classifier-onnx"


def main() -> None:
    if not (MODEL_DIR / "model.safetensors").exists():
        print(f"No trained PyTorch model found at {MODEL_DIR}. Run src/training/train.py first.")
        return

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_DIR} to ONNX format...")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.save_pretrained(ONNX_DIR)
    tokenizer.save_pretrained(ONNX_DIR)

    print("Quantizing to INT8 (reduces size/memory ~4x)...")
    quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_DIR, quantization_config=qconfig)

    # The quantizer writes "model_quantized.onnx" alongside the original
    # "model.onnx" -- replace the original with the quantized version so
    # our code (which always looks for "model.onnx") picks it up, and
    # remove the now-redundant full-precision copy.
    (ONNX_DIR / "model.onnx").unlink()
    (ONNX_DIR / "model_quantized.onnx").rename(ONNX_DIR / "model.onnx")

    # label_map.json is specific to this project, not a standard
    # transformers/optimum file -- copy it alongside manually.
    shutil.copy(MODEL_DIR / "label_map.json", ONNX_DIR / "label_map.json")

    print(f"\nSaved quantized ONNX model to {ONNX_DIR}")
    for f in sorted(ONNX_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
