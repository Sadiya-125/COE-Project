"""EfficientNet-B0 model with custom classification head for HPAFL.

Provides factory functions for model creation, BatchNorm layer identification
(for FedBN exclusion), backbone freezing, and parameter counting.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from config import HAPFLConfig

logger = logging.getLogger(__name__)


def get_model(config: HAPFLConfig) -> nn.Module:
    """Load EfficientNet-B0 with a custom multi-class classification head.

    Architecture:
        EfficientNet-B0 backbone (pretrained on ImageNet)
        → AdaptiveAvgPool (1280-dim features)
        → Linear(1280, 512) → BatchNorm1d(512) → ReLU → Dropout(dropout_rate)
        → Linear(512, num_classes)

    BN layers are identifiable via get_bn_layer_names() for FedBN exclusion.

    Args:
        config: HPAFL configuration with model_name, pretrained, dropout_rate,
                hidden_dim, and num_classes attributes.

    Returns:
        Configured nn.Module ready for training or federated use.

    Raises:
        ImportError: If efficientnet-pytorch is not installed.
    """
    try:
        from efficientnet_pytorch import EfficientNet
        # efficientnet-pytorch uses hyphens: 'efficientnet-b0', not underscores
        model_name = config.model_name.replace("_", "-")
        if config.pretrained:
            backbone = EfficientNet.from_pretrained(model_name)
        else:
            backbone = EfficientNet.from_name(model_name)
        # Replace the final FC layer with an identity so we get raw features
        in_features = backbone._fc.in_features
        backbone._fc = nn.Identity()
    except ImportError:
        # Fallback to torchvision EfficientNet
        logger.warning(
            "efficientnet-pytorch not found; falling back to torchvision EfficientNet"
        )
        import torchvision.models as tv_models
        weights = "IMAGENET1K_V1" if config.pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    model = _HPAFLEfficientNet(
        backbone=backbone,
        in_features=in_features,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
    )
    logger.info(
        "Built EfficientNet-B0 model: %s",
        count_parameters(model),
    )
    return model


class _HPAFLEfficientNet(nn.Module):
    """EfficientNet-B0 backbone with HPAFL classification head.

    Args:
        backbone: Pre-configured EfficientNet backbone (output = 1280-d vector).
        in_features: Feature dimensionality from backbone.
        hidden_dim: Hidden layer size in the classification head.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),  # inplace=True breaks Opacus backward hooks
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (N, 3, H, W).

        Returns:
            Logit tensor of shape (N, num_classes).
        """
        features = self.backbone(x)
        # Handle tuple output from some EfficientNet versions
        if isinstance(features, tuple):
            features = features[0]
        # Flatten if needed
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        return self.classifier(features)


def get_bn_layer_names(model: nn.Module) -> List[str]:
    """Return parameter names for all normalisation layers (BN and GN) for FedBN exclusion.

    Includes BatchNorm1d/2d/3d and GroupNorm. GroupNorm is included because
    Opacus replaces BatchNorm with GroupNorm before DP training, and FedBN
    should exclude both from aggregation (local normalisation statistics).

    Args:
        model: The neural network module.

    Returns:
        List of fully-qualified parameter name strings belonging to norm layers.
    """
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm
    )
    bn_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, norm_types):
            for param_name, _ in module.named_parameters():
                full_name = f"{name}.{param_name}" if name else param_name
                bn_names.append(full_name)
    return bn_names


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the EfficientNet backbone, keeping the classifier trainable.

    Args:
        model: The HPAFL EfficientNet model.
        freeze: True to freeze backbone parameters, False to unfreeze.
    """
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        status = "frozen" if freeze else "unfrozen"
        logger.info("Backbone %s. Classifier remains trainable.", status)
    else:
        logger.warning("Model has no 'backbone' attribute; no parameters were frozen.")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable, and frozen parameter counts.

    Args:
        model: Any nn.Module.

    Returns:
        Dict with keys 'total', 'trainable', 'frozen'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
