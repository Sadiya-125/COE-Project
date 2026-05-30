"""Tests for model architecture and focal loss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from config import HAPFLConfig
from models.efficientnet_model import get_model, get_bn_layer_names, count_parameters
from models.focal_loss import FocalLoss


@pytest.fixture(scope="module")
def cfg():
    c = HAPFLConfig()
    c.pretrained = False
    return c


@pytest.fixture(scope="module")
def model(cfg):
    return get_model(cfg)


def test_model_output_shape(model, cfg):
    """Forward pass must return (batch_size, num_classes) tensor."""
    batch_size = 4
    dummy = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (batch_size, cfg.num_classes), (
        f"Expected output shape ({batch_size}, {cfg.num_classes}), got {out.shape}"
    )


def test_bn_layer_detection(model):
    """get_bn_layer_names() must return a non-empty list."""
    bn_names = get_bn_layer_names(model)
    assert len(bn_names) > 0, "No BatchNorm layers detected — check model architecture."


def test_focal_loss_values(cfg):
    """FocalLoss should be positive and lower than CrossEntropy for easy examples."""
    import torch.nn.functional as F

    batch_size = 8
    num_classes = cfg.num_classes

    # Easy examples: one class has very high logit
    logits = torch.zeros(batch_size, num_classes)
    logits[:, 0] = 10.0  # Very confident about class 0
    labels = torch.zeros(batch_size, dtype=torch.long)  # All class 0

    fl = FocalLoss(gamma=2.0)
    ce = FocalLoss(gamma=0.0)  # gamma=0 reduces to cross-entropy

    focal_loss = fl(logits, labels)
    ce_loss = ce(logits, labels)

    assert focal_loss.item() > 0, "FocalLoss must be positive."
    assert focal_loss.item() < ce_loss.item(), (
        f"FocalLoss ({focal_loss.item():.6f}) should be < CE ({ce_loss.item():.6f}) "
        "for easy (high-confidence correct) examples when gamma > 0."
    )


def test_count_parameters(model):
    """count_parameters returns a dict with expected keys."""
    result = count_parameters(model)
    assert "total" in result
    assert "trainable" in result
    assert "frozen" in result
    assert result["total"] > 0
    assert result["trainable"] <= result["total"]
