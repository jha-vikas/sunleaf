"""Shared utilities for image preprocessing and model evaluation."""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    """Load a single image and return a batch tensor ready for inference."""
    img = Image.open(path).convert("RGB")
    transform = get_inference_transform(image_size)
    return transform(img).unsqueeze(0)


def predict_top_k(model: torch.nn.Module, image_tensor: torch.Tensor,
                  labels: list[str], k: int = 3) -> list[dict]:
    """Run inference and return top-k predictions with labels and confidence."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k, dim=1)

    results = []
    for i in range(k):
        idx = top_indices[0][i].item()
        results.append({
            "label": labels[idx] if idx < len(labels) else f"class_{idx}",
            "confidence": round(top_probs[0][i].item(), 4),
        })
    return results


def load_labels(path: str) -> list[str]:
    """Load label file (one label per line)."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
