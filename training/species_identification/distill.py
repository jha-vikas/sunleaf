"""
Knowledge distillation: ViT teacher -> MobileNetV2 student.

Downloads the pre-trained ViT teacher model (dima806/house-plant-image-detection)
from HuggingFace, downloads the houseplant dataset, and trains a compact
MobileNetV2 student model using knowledge distillation.

The student learns from both:
  - Soft labels (teacher's probability distribution, temperature-scaled)
  - Hard labels (ground truth class labels)

Usage (run on Mac M3 Max or machine with GPU):
    python training/species_identification/distill.py

Output:
    training/species_identification/checkpoints/best_model.pth
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

from dataset import get_dataloaders

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Distillation hyperparameters
TEMPERATURE = 3.0
ALPHA = 0.7          # Weight for soft (teacher) loss; (1 - ALPHA) for hard (CE) loss
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# HuggingFace teacher model
TEACHER_REPO = "dima806/house-plant-image-detection"


def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_teacher(num_classes: int, device: torch.device) -> nn.Module:
    """Load the pre-trained ViT teacher model from HuggingFace."""
    print(f"Loading teacher model from {TEACHER_REPO}...")
    teacher = ViTForImageClassification.from_pretrained(TEACHER_REPO)
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze teacher; we only need forward pass
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params")
    return teacher


def build_student(num_classes: int) -> nn.Module:
    """Build MobileNetV2 student with pretrained ImageNet backbone."""
    student = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    student.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(student.last_channel, num_classes),
    )
    param_count = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student model: MobileNetV2 with {param_count:.1f}M params")
    return student


class DistillationLoss(nn.Module):
    """Combined distillation loss: soft (KL divergence) + hard (cross entropy)."""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        distill_loss *= self.temperature ** 2

        hard_loss = self.ce_loss(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss


def get_teacher_logits(teacher, images, device):
    """
    Get teacher predictions. The HuggingFace ViT model expects its own
    preprocessing, but since we already normalized with ImageNet stats
    (which ViT also uses), we can pass images directly.
    """
    with torch.no_grad():
        outputs = teacher(pixel_values=images)
        return outputs.logits


@torch.no_grad()
def evaluate(student, val_loader, device):
    """Compute validation accuracy."""
    student.eval()
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = student(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


def train(dataset_path: Path | None = None):
    """Run the full distillation training loop."""
    device = get_device()

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        dataset_path=dataset_path,
    )
    num_classes = len(class_names)

    # Load teacher and build student
    teacher = load_teacher(num_classes, device)
    student = build_student(num_classes).to(device)

    # Verify teacher and student class counts match
    teacher_classes = teacher.config.num_labels
    if teacher_classes != num_classes:
        print(f"WARNING: Teacher has {teacher_classes} classes, dataset has {num_classes}.")
        print("Distillation will use dataset class count. Teacher outputs will be sliced/padded.")

    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    # Save class names for later use
    labels_path = Path(__file__).resolve().parent.parent.parent / "models" / "labels" / "species_labels.txt"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved {num_classes} class labels to {labels_path}")

    print(f"\n{'='*60}")
    print(f"Starting distillation: {EPOCHS} epochs")
    print(f"Temperature={TEMPERATURE}, Alpha={ALPHA}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            teacher_logits = get_teacher_logits(teacher, images, device)
            student_logits = student(images)

            # Handle class count mismatch if needed
            if teacher_logits.shape[1] != student_logits.shape[1]:
                min_classes = min(teacher_logits.shape[1], student_logits.shape[1])
                teacher_logits = teacher_logits[:, :min_classes]
                student_logits_for_loss = student_logits[:, :min_classes]
            else:
                student_logits_for_loss = student_logits

            loss = criterion(student_logits_for_loss, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start

        val_acc = evaluate(student, val_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.0f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "num_classes": num_classes,
                "class_names": class_names,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
            print(f"  -> New best model saved (acc: {val_acc:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation accuracy: {best_acc:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"{'='*60}")

    return student, class_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distill ViT teacher into MobileNetV2 student")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to dataset root (if already downloaded)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    train(dataset_path=dataset_path)
