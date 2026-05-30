"""Focal Loss implementation for addressing class imbalance.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    When gamma=0 and alpha=None, this reduces to standard CrossEntropyLoss.
    Higher gamma down-weights easy examples, focusing training on hard cases.

    Args:
        gamma: Focusing parameter. 0 = standard cross entropy. Default: 2.0
        alpha: Class weights tensor of shape (num_classes,). None means uniform.
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got '{reduction}'")
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape (N, C).
            targets: Integer class labels of shape (N,).

        Returns:
            Scalar loss (or per-sample tensor if reduction='none').

        Raises:
            ValueError: If logits and targets have incompatible shapes.
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2-D (N, C); got shape {logits.shape}")
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1-D (N,); got shape {targets.shape}")

        # log_softmax for numerical stability
        log_prob = F.log_softmax(logits, dim=-1)           # (N, C)
        prob = torch.exp(log_prob)                         # (N, C)

        # Gather predicted probability of the true class
        log_pt = log_prob.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt = prob.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)          # (N,)

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma            # (N,)

        # Class weight (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                  # (N,)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt                  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
