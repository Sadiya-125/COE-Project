"""HAM10000 PyTorch Dataset with augmentation and class-balanced sampling.

Provides train/validation dataset splits with configurable transforms and
a WeightedRandomSampler to handle severe class imbalance.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import CLASS_NAMES, DX_TO_CLASS, HOSPITAL_IDS, HAPFLConfig

logger = logging.getLogger(__name__)


class HAM10000Dataset(Dataset):
    """HAM10000 dermoscopic image dataset with configurable transforms.

    Args:
        data_root: Root directory containing HAM10000_metadata.csv and image folders.
        indices: List of integer row indices into the metadata DataFrame.
        split: 'train' for full augmentation or 'val' for resize+crop+normalize only.
        config: HPAFL configuration object; uses image_size if provided.
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root: str,
        indices: List[int],
        split: str = "train",
        config: Optional[HAPFLConfig] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.indices = indices
        self.split = split
        self.config = config or HAPFLConfig()
        self.image_size = self.config.image_size

        self.df = self._load_metadata()
        self.subset = self.df.iloc[indices].reset_index(drop=True)
        self.transform = self._build_transform()
        self._image_dirs = self._find_image_dirs()

    def _load_metadata(self) -> pd.DataFrame:
        """Load and label-encode the HAM10000 metadata CSV.

        Returns:
            DataFrame with 'label' integer column.

        Raises:
            FileNotFoundError: If metadata CSV does not exist.
        """
        csv_path = self.data_root / "HAM10000_metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df["class_code"] = df["dx"].map(DX_TO_CLASS)
        df["label"] = df["class_code"].map(
            {name: idx for idx, name in enumerate(CLASS_NAMES)}
        )
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df

    def _find_image_dirs(self) -> List[Path]:
        """Discover HAM10000 image part directories inside data_root.

        Returns:
            List of Path objects pointing to image directories.
        """
        dirs = []
        for candidate in sorted(self.data_root.iterdir()):
            if candidate.is_dir() and "images" in candidate.name.lower():
                dirs.append(candidate)
        if not dirs:
            # Fallback: use data_root itself
            dirs = [self.data_root]
        return dirs

    def _find_image(self, image_id: str) -> Path:
        """Locate a JPEG image file across all image part directories.

        Args:
            image_id: ISIC image identifier string.

        Returns:
            Path to the image file.

        Raises:
            FileNotFoundError: If image is not found in any known directory.
        """
        for img_dir in self._image_dirs:
            path = img_dir / f"{image_id}.jpg"
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Image {image_id}.jpg not found in {[str(d) for d in self._image_dirs]}"
        )

    def _build_transform(self) -> transforms.Compose:
        """Build the torchvision transform pipeline for train or val split.

        Supports configurable augmentation strength via config.augmentation_strength:
        - "low": Minimal augmentation (flip, small rotation)
        - "medium": Balanced augmentation (default)
        - "high": Aggressive augmentation (all transforms with stronger parameters)

        Returns:
            Composed transform pipeline.
        """
        if self.split == "train":
            # Get augmentation strength and whether augmentation is enabled
            use_aug = getattr(self.config, 'use_augmentation', True)
            aug_strength = getattr(self.config, 'augmentation_strength', 'high')

            if not use_aug:
                # Minimal transforms only (normalize)
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self._MEAN, std=self._STD),
                    ]
                )
            elif aug_strength == "low":
                # Minimal augmentation
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.RandomHorizontalFlip(p=0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self._MEAN, std=self._STD),
                    ]
                )
            elif aug_strength == "medium":
                # Balanced augmentation
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                        ),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self._MEAN, std=self._STD),
                    ]
                )
            else:  # "high" or default
                # Aggressive augmentation
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.RandomHorizontalFlip(p=0.6),
                        transforms.RandomVerticalFlip(p=0.4),
                        transforms.RandomRotation(degrees=25),
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                        ),
                        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self._MEAN, std=self._STD),
                    ]
                )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self._MEAN, std=self._STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fetch a single (image, label) pair.

        Args:
            idx: Index into the subset.

        Returns:
            Tuple of (transformed image tensor, integer class label).
        """
        row = self.subset.iloc[idx]
        label = int(row["label"])
        image_id = row["image_id"]

        img_path = self._find_image(image_id)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for WeightedRandomSampler.

        Returns:
            Float tensor of shape (num_classes,) with per-class sample weights.
        """
        labels = self.subset["label"].values
        class_counts = np.bincount(labels, minlength=self.config.num_classes).astype(float)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler balancing under-represented classes.

        Returns:
            WeightedRandomSampler instance.
        """
        cw = self.class_weights
        labels = self.subset["label"].values.copy()  # ensure writable for tensor conversion
        sample_weights = cw[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def create_dataloaders(
    hospital_id: str,
    config: HAPFLConfig,
    val_fraction: float = 0.15,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for a specific hospital.

    Args:
        hospital_id: Hospital identifier ('A', 'B', or 'C').
        config: HPAFL configuration object.
        val_fraction: Fraction of hospital data held out for validation.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        FileNotFoundError: If the hospital partition file does not exist.
    """
    # Partitions live in output_root (writable), not data_root (read-only on Kaggle)
    partition_path = config.partition_dir / f"hospital_{hospital_id}_indices.json"

    if not partition_path.exists():
        raise FileNotFoundError(
            f"Partition file not found: {partition_path}. "
            "Run data/prepare_data.py first."
        )

    with open(partition_path) as f:
        all_indices = json.load(f)

    rng = np.random.default_rng(42 + ord(hospital_id))
    all_indices = np.array(all_indices)
    rng.shuffle(all_indices)

    n_val = max(1, int(len(all_indices) * val_fraction))
    val_indices = all_indices[:n_val].tolist()
    train_indices = all_indices[n_val:].tolist()

    train_ds = HAM10000Dataset(
        data_root=config.data_root,
        indices=train_indices,
        split="train",
        config=config,
    )
    val_ds = HAM10000Dataset(
        data_root=config.data_root,
        indices=val_indices,
        split="val",
        config=config,
    )

    use_cuda = torch.cuda.is_available()
    nw = getattr(config, "num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_ds.get_sampler(),
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=nw > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=nw > 0,
    )

    logger.info(
        "Hospital %s — train: %d samples, val: %d samples",
        hospital_id,
        len(train_ds),
        len(val_ds),
    )
    return train_loader, val_loader
