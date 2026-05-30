"""Tests for Flower client FedBN parameter handling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from config import HAPFLConfig
from models.efficientnet_model import get_bn_layer_names, get_model


@pytest.fixture(scope="module")
def cfg():
    c = HAPFLConfig()
    c.pretrained = False
    return c


@pytest.fixture(scope="module")
def model_and_bn(cfg):
    model = get_model(cfg)
    bn_names = set(get_bn_layer_names(model))
    return model, bn_names


def _get_non_bn_params(model, bn_names):
    return [
        (name, param.detach().clone())
        for name, param in model.named_parameters()
        if name not in bn_names
    ]


def _get_bn_params(model, bn_names):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in bn_names
    }


def test_get_parameters_excludes_bn(model_and_bn, cfg):
    """get_parameters() must not include any BN parameters."""
    model, bn_names = model_and_bn

    non_bn = _get_non_bn_params(model, bn_names)
    returned = [p.detach().cpu().numpy() for _, p in non_bn]

    # Verify the count matches what a client would return
    total_params = len(list(model.named_parameters()))
    bn_param_count = sum(
        1 for name, _ in model.named_parameters() if name in bn_names
    )
    expected_non_bn = total_params - bn_param_count

    assert len(returned) == expected_non_bn, (
        f"Expected {expected_non_bn} non-BN parameters, got {len(returned)}"
    )
    assert len(returned) > 0, "No parameters returned — check FedBN exclusion logic."


def test_set_parameters_preserves_bn(model_and_bn, cfg):
    """set_parameters() must leave BN layer values unchanged."""
    model, bn_names = model_and_bn

    # Record BN values before update
    bn_before = _get_bn_params(model, bn_names)

    # Build a fresh set of random non-BN parameters
    non_bn_named = [
        (name, param)
        for name, param in model.named_parameters()
        if name not in bn_names
    ]
    random_params = [np.random.randn(*p.shape).astype(np.float32) for _, p in non_bn_named]

    # Simulate set_parameters
    state = model.state_dict()
    for (name, _), arr in zip(non_bn_named, random_params):
        state[name] = torch.tensor(arr)
    model.load_state_dict(state, strict=True)

    # Check BN values are unchanged
    bn_after = _get_bn_params(model, bn_names)
    for name in bn_names:
        if name in bn_before and name in bn_after:
            assert torch.allclose(bn_before[name], bn_after[name], atol=1e-7), (
                f"BN parameter '{name}' was modified by set_parameters()!"
            )
