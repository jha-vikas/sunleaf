"""
Dataset loader for houseplant species identification.

Downloads the house plant image dataset from Kaggle and prepares
PyTorch DataLoaders for training and validation.

Dataset: https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import get_training_transform, get_inference_transform


DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def download_dataset() -> Path:
    """Download the house plant dataset via kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("kacpergregorowicz/house-plant-species")
        print(f"Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species")
        print("2. Download and extract to: training/species_identification/dataset/")
        print("3. Ensure structure is: dataset/<species_name>/<images>.jpg")
        raise


def find_image_root(base_path: Path) -> Path:
    """Find the directory that contains class subdirectories."""
    # Kaggle downloads sometimes nest the data one level deep
    if any((base_path / d).is_dir() and not d.startswith(".") for d in 
           [p.name for p in base_path.iterdir() if p.is_dir()]):
        subdirs = [p for p in base_path.iterdir() 
                   if p.is_dir() and not p.name.startswith(".")]
        # Check if subdirs contain images directly (this is the image root)
        for sd in subdirs[:3]:
            files = list(sd.glob("*.jpg")) + list(sd.glob("*.png")) + list(sd.glob("*.jpeg"))
            if files:
                return base_path
        # Otherwise recurse one level deeper
        for sd in subdirs:
            result = find_image_root(sd)
            if result != sd:
                return result
            # Check if this subdir's children have images
            for ssd in sd.iterdir():
                if ssd.is_dir():
                    files = list(ssd.glob("*.jpg")) + list(ssd.glob("*.png"))
                    if files:
                        return sd
    return base_path


def get_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.15,
    num_workers: int = 4,
    dataset_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Create training and validation DataLoaders.

    Returns:
        train_loader, val_loader, class_names
    """
    if dataset_path is None:
        dataset_path = download_dataset()

    image_root = find_image_root(dataset_path)
    print(f"Using image root: {image_root}")

    full_dataset = datasets.ImageFolder(root=str(image_root))
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes, {len(full_dataset)} images total")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_transform = get_training_transform(image_size)
    val_transform = get_inference_transform(image_size)

    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_loader = DataLoader(
        TransformSubset(train_subset, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TransformSubset(val_subset, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {train_size} images, Val: {val_size} images")
    return train_loader, val_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders(batch_size=4)
    print(f"\nClasses ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i:3d}: {name}")

    batch = next(iter(train_loader))
    print(f"\nSample batch shape: {batch[0].shape}")
