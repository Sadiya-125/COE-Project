"""Differentially private training wrapper using Opacus PrivacyEngine.

Implements DP-SGD (Abadi et al., CCS 2016) with per-round epsilon tracking
and a CSV budget logger for post-hoc analysis.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import HAPFLConfig

logger = logging.getLogger(__name__)


class DPTrainer:
    """Wraps Opacus PrivacyEngine for differentially private local training.

    Attaches to a model + optimizer + dataloader triplet and tracks cumulative
    (epsilon, delta) privacy budget across all training steps.

    Args:
        config: HPAFL configuration with DP hyperparameters.
    """

    def __init__(self, config: HAPFLConfig) -> None:
        self.config = config
        self._privacy_engine = None
        self._epsilon: float = 0.0
        self._attached: bool = False

    def attach(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
    ) -> Tuple[nn.Module, Optimizer, DataLoader]:
        """Attach Opacus PrivacyEngine to model, optimizer, and dataloader.

        Must be called before any training steps. Returns the wrapped
        (dp_model, dp_optimizer, dp_dataloader) that must be used in place
        of the originals during training.

        Args:
            model: The nn.Module to make differentially private.
            optimizer: The optimiser to wrap.
            dataloader: The DataLoader to wrap.

        Returns:
            Tuple of (dp_model, dp_optimizer, dp_dataloader).

        Raises:
            ImportError: If Opacus is not installed.
            RuntimeError: If PrivacyEngine cannot be attached.
        """
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
        except ImportError as exc:
            raise ImportError(
                "Opacus is required for DP training. Install with: pip install opacus"
            ) from exc

        # Model must already have BatchNorm replaced with GroupNorm before this
        # call (done at model initialisation time in run_hpafl.py via
        # ModuleValidator.fix()). Validate here to give a clear error if not.
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            raise RuntimeError(
                f"Model is not DP-compatible ({len(errors)} errors). "
                "Call ModuleValidator.fix(model) before creating the optimizer."
            )

        self._privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_dataloader = self._privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            target_epsilon=self.config.target_epsilon,
            target_delta=self.config.target_delta,
            epochs=self.config.local_epochs,
            max_grad_norm=self.config.max_grad_norm,
        )
        self._attached = True
        logger.info(
            "DP-SGD attached — target ε=%.2f, δ=%.1e, max_grad_norm=%.2f",
            self.config.target_epsilon,
            self.config.target_delta,
            self.config.max_grad_norm,
        )
        return dp_model, dp_optimizer, dp_dataloader

    def get_epsilon(self) -> float:
        """Return the current cumulative epsilon privacy spend.

        Returns:
            Current epsilon value; returns cached _epsilon (default 0.0) if
            engine has not been attached yet.
        """
        if self._privacy_engine is None or not self._attached:
            return self._epsilon
        try:
            eps = self._privacy_engine.get_epsilon(self.config.target_delta)
            eps_val = float(eps)
            # PRV accountant returns NaN before any training steps — keep cached value
            if eps_val == eps_val:  # NaN check (NaN != NaN)
                self._epsilon = eps_val
        except Exception:
            pass  # keep cached _epsilon; don't spam log before first step
        return self._epsilon

    def is_budget_exhausted(self) -> bool:
        """Return True if the current epsilon spend meets or exceeds the target.

        Returns:
            True if epsilon >= target_epsilon.
        """
        return self.get_epsilon() >= self.config.target_epsilon

    def get_privacy_report(self) -> Dict[str, float]:
        """Return a summary dictionary of the current privacy state.

        Returns:
            Dict with keys: epsilon, delta, noise_multiplier, max_grad_norm,
            budget_remaining (target_epsilon - current_epsilon).
        """
        eps = self.get_epsilon()
        return {
            "epsilon": eps,
            "delta": self.config.target_delta,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm": self.config.max_grad_norm,
            "budget_remaining": max(0.0, self.config.target_epsilon - eps),
        }


class PrivacyBudgetLogger:
    """Appends per-round (hospital, round, epsilon, delta) rows to a CSV file.

    Args:
        results_dir: Directory where privacy_budget.csv will be written.
    """

    _FIELDNAMES = ["round", "hospital_id", "epsilon", "delta", "budget_remaining"]

    def __init__(self, results_dir: str) -> None:
        self.path = Path(results_dir) / "privacy_budget.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writeheader()
        logger.info("PrivacyBudgetLogger writing to %s", self.path)

    def log(
        self,
        round_num: int,
        hospital_id: str,
        epsilon: float,
        delta: float,
        target_epsilon: float,
    ) -> None:
        """Append one row to the CSV log.

        Args:
            round_num: Current FL round number.
            hospital_id: Hospital identifier string.
            epsilon: Cumulative epsilon spend this round.
            delta: Target delta value.
            target_epsilon: Maximum allowed epsilon.
        """
        row = {
            "round": round_num,
            "hospital_id": hospital_id,
            "epsilon": round(epsilon, 6),
            "delta": delta,
            "budget_remaining": round(max(0.0, target_epsilon - epsilon), 6),
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerow(row)
