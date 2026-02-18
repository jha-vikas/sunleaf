"""
Tests to validate TFLite models after conversion/export.

These tests check that:
1. Model files exist and are reasonable sizes
2. Labels files match expected class counts
3. PyTorch models (pre-conversion) produce valid output shapes

Run after model conversion/export:
    pytest tests/test_models.py -v
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LABELS_DIR = MODELS_DIR / "labels"
DATA_DIR = PROJECT_ROOT / "data"


class TestDiseaseModel:
    """Tests for the disease detection model."""

    def test_mobilenetv2_output_shape(self):
        """Verify MobileNetV2 with 38-class head produces correct output."""
        num_classes = 38
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (1, num_classes)
        probs = torch.nn.functional.softmax(output, dim=1)
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_disease_labels_file(self):
        """Check disease labels file exists and has 38 classes."""
        labels_path = LABELS_DIR / "disease_labels.txt"
        if labels_path.exists():
            with open(labels_path) as f:
                labels = [l.strip() for l in f if l.strip()]
            assert len(labels) == 38, f"Expected 38 labels, got {len(labels)}"

    def test_disease_tflite_exists(self):
        """Check TFLite model exists (only after conversion)."""
        tflite_path = MODELS_DIR / "plant_disease_v1.tflite"
        if tflite_path.exists():
            size_mb = tflite_path.stat().st_size / (1024 * 1024)
            assert 1 < size_mb < 50, f"Unexpected model size: {size_mb:.1f} MB"


class TestSpeciesModel:
    """Tests for the species identification model."""

    def test_mobilenetv2_student_output_shape(self):
        """Verify MobileNetV2 student with 47-class head produces correct output."""
        num_classes = 47
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes),
        )
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (1, num_classes)

    def test_species_tflite_exists(self):
        """Check species TFLite model exists (only after training)."""
        tflite_path = MODELS_DIR / "plant_species.tflite"
        if tflite_path.exists():
            size_mb = tflite_path.stat().st_size / (1024 * 1024)
            assert 1 < size_mb < 50, f"Unexpected model size: {size_mb:.1f} MB"


class TestKnowledgeBases:
    """Tests for the curated knowledge base files."""

    def test_disease_remedies_valid_json(self):
        path = DATA_DIR / "disease_remedies.json"
        assert path.exists(), "disease_remedies.json not found"
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 38, f"Expected 38 entries, got {len(data)}"

    def test_disease_remedies_structure(self):
        path = DATA_DIR / "disease_remedies.json"
        with open(path) as f:
            data = json.load(f)
        for key, entry in data.items():
            assert "display_name" in entry, f"Missing display_name in {key}"
            assert "symptoms" in entry, f"Missing symptoms in {key}"
            assert "home_remedies" in entry, f"Missing home_remedies in {key}"
            assert "prevention" in entry, f"Missing prevention in {key}"

    def test_plant_care_valid_json(self):
        path = DATA_DIR / "plant_care.json"
        assert path.exists(), "plant_care.json not found"
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 47, f"Expected 47 entries, got {len(data)}"

    def test_plant_care_structure(self):
        path = DATA_DIR / "plant_care.json"
        with open(path) as f:
            data = json.load(f)
        for key, entry in data.items():
            assert "light" in entry, f"Missing light in {key}"
            assert "water" in entry, f"Missing water in {key}"
            assert "humidity" in entry, f"Missing humidity in {key}"
            assert "temperature" in entry, f"Missing temperature in {key}"
