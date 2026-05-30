"""Tests for DP trainer and privacy budget tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import HAPFLConfig
from privacy.dp_trainer import DPTrainer


@pytest.fixture
def cfg():
    c = HAPFLConfig()
    c.local_epochs = 2
    c.batch_size = 8
    c.target_epsilon = 4.0
    c.target_delta = 1e-5
    c.noise_multiplier = 1.1
    c.max_grad_norm = 1.0
    return c


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 7))


@pytest.fixture
def dummy_loader():
    x = torch.randn(32, 16)
    y = torch.randint(0, 7, (32,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8)


def test_dp_trainer_attaches(cfg, simple_model, dummy_loader):
    """DPTrainer.attach() must succeed and return wrapped triplet."""
    trainer = DPTrainer(cfg)
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    try:
        dp_model, dp_opt, dp_loader = trainer.attach(simple_model, optimizer, dummy_loader)
        assert dp_model is not None
        assert dp_opt is not None
        assert dp_loader is not None
    except ImportError:
        pytest.skip("Opacus not installed — skipping DP attach test.")


def test_epsilon_increases(cfg, simple_model, dummy_loader):
    """Epsilon must increase monotonically with training steps."""
    try:
        import opacus
    except ImportError:
        pytest.skip("Opacus not installed.")

    trainer = DPTrainer(cfg)
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    dp_model, dp_opt, dp_loader = trainer.attach(simple_model, optimizer, dummy_loader)

    criterion = nn.CrossEntropyLoss()
    epsilons = []

    dp_model.train()
    for images, labels in dp_loader:
        dp_opt.zero_grad()
        out = dp_model(images)
        loss = criterion(out, labels)
        loss.backward()
        dp_opt.step()
        eps = trainer.get_epsilon()
        epsilons.append(eps)

    assert epsilons[-1] >= epsilons[0], (
        "Epsilon should be non-decreasing across training steps."
    )


def test_budget_exhausted(cfg):
    """is_budget_exhausted() must return True when epsilon >= target."""
    trainer = DPTrainer(cfg)
    trainer._epsilon = cfg.target_epsilon + 0.1  # Manually set above target
    assert trainer.is_budget_exhausted() is True, (
        "is_budget_exhausted() should return True when epsilon >= target_epsilon."
    )
