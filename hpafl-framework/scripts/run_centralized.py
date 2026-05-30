"""Centralised training baseline on the full HAM10000 dataset.

Trains EfficientNet-B0 without any federation as the performance ceiling.
Saves results to results/centralized.json.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CLASS_NAMES, HAPFLConfig
from data.dataset import HAM10000Dataset
from evaluation.evaluate import evaluate_model, plot_confusion_matrix
from models.efficientnet_model import get_model
from models.focal_loss import FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_centralized(config: HAPFLConfig) -> None:
    """Train a centralised model on all HAM10000 data and save results.

    Args:
        config: HPAFL configuration object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("Centralised training on %s", device)

    # Use all indices for training
    import pandas as pd
    from config import DX_TO_CLASS

    csv_path = Path(config.data_root) / "HAM10000_metadata.csv"
    if not csv_path.exists():
        logger.error("Metadata CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df["class_code"] = df["dx"].map(DX_TO_CLASS)
    df["label"] = df["class_code"].map(
        {name: idx for idx, name in enumerate(CLASS_NAMES)}
    )
    df = df.dropna(subset=["label"])
    all_indices = list(range(len(df)))

    # 85/15 train/val split
    rng = np.random.default_rng(42)
    rng.shuffle(all_indices)
    n_val = int(0.15 * len(all_indices))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    train_ds = HAM10000Dataset(config.data_root, train_indices, split="train", config=config)
    val_ds = HAM10000Dataset(config.data_root, val_indices, split="val", config=config)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=train_ds.get_sampler(),
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2,
    )

    model = get_model(config).to(device)
    alpha = train_ds.class_weights.to(device)
    criterion = FocalLoss(gamma=config.focal_gamma, alpha=alpha)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    n_epochs = config.num_rounds  # Use same budget as FL
    best_f1 = 0.0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_model(model, val_loader, CLASS_NAMES, device)
        logger.info(
            "Epoch %d/%d | Loss: %.4f | Acc: %.4f | F1: %.4f",
            epoch + 1, n_epochs, total_loss / len(train_loader),
            metrics["accuracy"], metrics["f1_macro"],
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            ckpt = Path(config.checkpoint_dir) / "centralized_best.pt"
            torch.save(model.state_dict(), ckpt)

    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, CLASS_NAMES, device)
    final_metrics["epsilon"] = float("inf")  # No DP
    final_metrics["system"] = "centralized"

    out_path = Path(config.results_dir) / "centralized.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("Centralised results saved → %s", out_path)

    # Confusion matrix plot
    import numpy as np
    cm = np.array(final_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, CLASS_NAMES, "Centralised Model Confusion Matrix")
    logger.info("Done. Best F1: %.4f | Final Acc: %.4f", best_f1, final_metrics["accuracy"])


if __name__ == "__main__":
    cfg = HAPFLConfig()
    cfg.data_root = str(Path(__file__).resolve().parents[2] / "archive")
    run_centralized(cfg)
