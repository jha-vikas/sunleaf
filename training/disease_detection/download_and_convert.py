"""
Download pre-trained plant disease MobileNetV2 model from HuggingFace
and convert it to TFLite format for mobile deployment.

Model: Daksh159/plant-disease-mobilenetv2
- 38 disease classes across 14 crop species
- ~95% validation accuracy
- MobileNetV2 backbone

Usage:
    python training/disease_detection/download_and_convert.py

Output:
    models/plant_disease_v1.tflite
    models/labels/disease_labels.txt
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LABELS_DIR = MODELS_DIR / "labels"

HF_REPO = "Daksh159/plant-disease-mobilenetv2"

DISEASE_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(DISEASE_LABELS)


def build_mobilenetv2(num_classes: int) -> nn.Module:
    """Build MobileNetV2 with custom classifier head matching the pre-trained model."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def download_model() -> Path:
    """Download the pre-trained weights from HuggingFace."""
    print(f"Downloading model from {HF_REPO}...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="plant_disease_mobilenetv2.pth",
    )
    print(f"Downloaded to: {path}")
    return Path(path)


def load_pretrained_model(weights_path: Path) -> nn.Module:
    """Load the pre-trained weights into MobileNetV2."""
    model = build_mobilenetv2(NUM_CLASSES)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Handle different state_dict formats from HuggingFace uploads
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Try loading; if keys don't match exactly, try stripping module. prefix
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)

    model.eval()
    print(f"Loaded model with {NUM_CLASSES} classes")
    return model


def convert_to_tflite(model: nn.Module, output_path: Path) -> None:
    """Convert PyTorch model to TFLite using litert-torch."""
    try:
        import litert_torch
    except ImportError:
        print("ERROR: litert-torch not installed. Install with: uv pip install litert-torch")
        print("Falling back to ONNX export instead...")
        export_onnx_fallback(model, output_path.with_suffix(".onnx"))
        return

    print("Converting to TFLite via litert-torch...")
    sample_input = torch.randn(1, 3, 224, 224)

    tflite_model = litert_torch.converter.convert(
        model,
        sample_input,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_model.export(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"TFLite model saved to: {output_path} ({size_mb:.1f} MB)")


def export_onnx_fallback(model: nn.Module, output_path: Path) -> None:
    """Fallback: export to ONNX if litert-torch is not available."""
    print("Exporting to ONNX format...")
    sample_input = torch.randn(1, 3, 224, 224)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        input_names=["image"],
        output_names=["predictions"],
        dynamic_axes={"image": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=13,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved to: {output_path} ({size_mb:.1f} MB)")
    print("NOTE: Convert ONNX to TFLite using: pip install onnx2tf && onnx2tf -i model.onnx")


def save_labels(labels: list[str], output_path: Path) -> None:
    """Save labels to a text file (one per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels saved to: {output_path} ({len(labels)} classes)")


def quick_validate(model: nn.Module) -> None:
    """Quick sanity check: run inference on a random tensor."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    probs = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_idx = torch.max(probs, dim=1)
    print(f"Sanity check passed: predicted class {top_idx.item()} "
          f"({DISEASE_LABELS[top_idx.item()]}) with confidence {top_prob.item():.4f}")


def main():
    print("=" * 60)
    print("Plant Disease Detection Model - Download & Convert")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    weights_path = download_model()

    # Step 2: Load into MobileNetV2
    model = load_pretrained_model(weights_path)

    # Step 3: Quick validation
    quick_validate(model)

    # Step 4: Save labels
    save_labels(DISEASE_LABELS, LABELS_DIR / "disease_labels.txt")

    # Step 5: Convert to TFLite
    tflite_path = MODELS_DIR / "plant_disease_v1.tflite"
    convert_to_tflite(model, tflite_path)

    print("\n" + "=" * 60)
    print("Done! Files created:")
    print(f"  Model:  {tflite_path}")
    print(f"  Labels: {LABELS_DIR / 'disease_labels.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
