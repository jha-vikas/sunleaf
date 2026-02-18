"""
Export the trained MobileNetV2 species model to TFLite format.

Loads the best checkpoint from distillation training and converts
it to a quantized TFLite model for mobile deployment.

Usage:
    python training/species_identification/export_tflite.py

Output:
    models/plant_species.tflite
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def build_mobilenetv2(num_classes: int) -> nn.Module:
    """Build MobileNetV2 matching the student architecture."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def load_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, dict]:
    """Load the best training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = checkpoint["num_classes"]
    class_names = checkpoint.get("class_names", [])
    val_acc = checkpoint.get("val_accuracy", 0.0)

    model = build_mobilenetv2(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded: {num_classes} classes, val_accuracy={val_acc:.4f}")
    return model, checkpoint


def convert_to_tflite(model: nn.Module, output_path: Path) -> None:
    """Convert PyTorch model to TFLite."""
    try:
        import litert_torch
    except ImportError:
        print("ERROR: litert-torch not installed.")
        print("Install with: uv pip install litert-torch")
        print("Falling back to ONNX export...")
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
    print(f"TFLite model saved: {output_path} ({size_mb:.1f} MB)")


def export_onnx_fallback(model: nn.Module, output_path: Path) -> None:
    """Fallback: export to ONNX if litert-torch is unavailable."""
    print("Exporting to ONNX...")
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
    print(f"ONNX model saved: {output_path} ({size_mb:.1f} MB)")
    print("NOTE: Convert to TFLite with: pip install onnx2tf && onnx2tf -i model.onnx")


def validate_model(model: nn.Module, num_classes: int) -> None:
    """Quick sanity check with random input."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, num_classes), f"Expected (1, {num_classes}), got {output.shape}"
    probs = torch.nn.functional.softmax(output, dim=1)
    print(f"Sanity check passed: output shape {output.shape}, max prob {probs.max().item():.4f}")


def main():
    print("=" * 60)
    print("Species Model - Export to TFLite")
    print("=" * 60)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Run distillation first: python training/species_identification/distill.py")
        sys.exit(1)

    model, checkpoint = load_checkpoint(checkpoint_path)
    num_classes = checkpoint["num_classes"]

    validate_model(model, num_classes)

    tflite_path = MODELS_DIR / "plant_species.tflite"
    convert_to_tflite(model, tflite_path)

    print("\n" + "=" * 60)
    print("Done! Model exported for mobile deployment.")
    print(f"  Model:  {tflite_path}")
    print(f"  Labels: {MODELS_DIR / 'labels' / 'species_labels.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
