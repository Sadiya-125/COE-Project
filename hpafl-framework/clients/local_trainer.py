"""Local model training with DP-SGD and FocalLoss for HPAFL clients.

Manages per-round local training, validation, and privacy budget reporting
for a single hospital node.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import CLASS_NAMES, HAPFLConfig
from models.focal_loss import FocalLoss
from privacy.dp_trainer import DPTrainer

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Manages local model training with DP-SGD and FocalLoss.

    Trains for config.local_epochs epochs, applies DP-SGD via DPTrainer,
    and returns updated parameters with training metrics.

    Args:
        config: HPAFL configuration object.
        hospital_id: Identifier of this hospital (for logging).
        device: PyTorch device to run training on.
    """

    def __init__(
        self,
        config: HAPFLConfig,
        hospital_id: str = "A",
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.hospital_id = hospital_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: Optional[float] = None,
        use_dp: bool = True,
    ) -> Dict:
        """Run local training for one FL round.

        Args:
            model: The model to train (on CPU; moved to device internally).
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            learning_rate: Override LR from config if provided.
            use_dp: Whether to apply DP-SGD via Opacus.

        Returns:
            Dict with keys: loss, accuracy, f1_macro, epsilon,
            num_samples, epochs_trained.
        """
        lr = learning_rate or self.config.learning_rate
        model = model.to(self.device)

        # Build loss with class weights from training set
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, "class_weights"):
            alpha = train_dataset.class_weights.to(self.device)
        else:
            alpha = None
        criterion = FocalLoss(gamma=self.config.focal_gamma, alpha=alpha)

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        dp_trainer = DPTrainer(self.config)
        epsilon = 0.0

        # Clear GPU cache before DP-SGD attachment to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_dp:
            try:
                model, optimizer, train_loader = dp_trainer.attach(
                    model, optimizer, train_loader
                )
                logger.info(
                    "Hospital %s: DP-SGD attached.", self.hospital_id
                )
            except Exception as exc:
                logger.warning(
                    "Hospital %s: DP attach failed (%s). Falling back to standard SGD.",
                    self.hospital_id,
                    exc,
                )
                use_dp = False

        final_loss = 0.0
        for epoch in range(self.config.local_epochs):
            if use_dp and dp_trainer.is_budget_exhausted():
                logger.warning(
                    "Hospital %s: Privacy budget exhausted at epoch %d.",
                    self.hospital_id,
                    epoch,
                )
                break
            loss = self._train_epoch(model, train_loader, optimizer, criterion)
            final_loss = loss
            if use_dp:
                epsilon = dp_trainer.get_epsilon()
            logger.debug(
                "Hospital %s | Epoch %d/%d | Loss: %.4f | ε=%.4f",
                self.hospital_id,
                epoch + 1,
                self.config.local_epochs,
                loss,
                epsilon,
            )

        val_metrics = self._validate(model, val_loader)
        num_samples = len(train_loader.dataset)

        return {
            "loss": float(final_loss),
            "accuracy": val_metrics["accuracy"],
            "f1_macro": val_metrics["f1_macro"],
            "epsilon": epsilon,
            "num_samples": num_samples,
            "epochs_trained": self.config.local_epochs,
        }

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Run a single training epoch.

        Args:
            model: Model in training mode.
            loader: Training DataLoader.
            optimizer: Optimiser (potentially DP-wrapped).
            criterion: Loss function.

        Returns:
            Average loss over the epoch.
        """
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Clear GPU cache after each batch to free memory for next iteration
            # Critical for DP-SGD which has high per-sample gradient memory overhead
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / max(n_batches, 1)

    def _validate(self, model: nn.Module, loader: DataLoader) -> Dict:
        """Run a validation pass and compute accuracy and macro F1.

        Args:
            model: Model to evaluate.
            loader: Validation DataLoader.

        Returns:
            Dict with keys: accuracy, f1_macro, per_class_f1.
        """
        model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                logits = model(images)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        accuracy = float(np.mean(all_preds_arr == all_labels_arr))
        f1_macro = float(
            f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
        )
        per_class_f1 = f1_score(
            all_labels_arr,
            all_preds_arr,
            average=None,
            zero_division=0,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist()

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "per_class_f1": {
                CLASS_NAMES[i]: round(per_class_f1[i], 4)
                for i in range(len(CLASS_NAMES))
            },
        }


