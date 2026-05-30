"""Tests for adaptive aggregation and secure aggregation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from config import HAPFLConfig
from security.secure_agg import SecureAggregator


@pytest.fixture
def cfg():
    return HAPFLConfig()


@pytest.fixture
def secure_agg():
    return SecureAggregator(num_clients=3, threshold=2, seed=42)


def _make_dummy_update(n_layers=3, layer_size=10):
    return [np.random.randn(layer_size).astype(np.float64) for _ in range(n_layers)]


def test_secure_agg_correctness(secure_agg):
    """SecureAggregator.aggregate_with_round() must equal the true element-wise sum."""
    round_num = 1
    true_updates = {
        0: _make_dummy_update(),
        1: _make_dummy_update(),
        2: _make_dummy_update(),
    }
    true_sum = [
        sum(true_updates[i][k] for i in range(3))
        for k in range(len(true_updates[0]))
    ]

    masked = {
        i: secure_agg.mask_update(i, true_updates[i], round_num)
        for i in range(3)
    }
    recovered = secure_agg.aggregate_with_round(masked, round_num=round_num)

    for k in range(len(true_sum)):
        np.testing.assert_allclose(
            recovered[k], true_sum[k], atol=1e-6,
            err_msg=f"Layer {k}: recovered aggregate ≠ true sum"
        )


def test_adaptive_weights_sum_to_one(cfg):
    """Normalised adaptive weights must sum to 1.0."""
    from server.adaptive_agg import AdaptiveAggregationStrategy
    from security.secure_agg import SecureAggregator

    sec = SecureAggregator(num_clients=cfg.num_hospitals, threshold=2)
    strategy = AdaptiveAggregationStrategy(config=cfg, secure_aggregator=sec)

    weights_raw = {}
    total = 0.0
    for i, hosp in enumerate(["A", "B", "C"]):
        w, _ = strategy._compute_adaptive_weight(
            client_id=hosp,
            accuracy=0.7 + i * 0.05,
            num_samples=1000 + i * 100,
            max_samples=1200,
            round_num=1,
        )
        weights_raw[hosp] = w
        total += w

    norm = {k: v / total for k, v in weights_raw.items()}
    total_norm = sum(norm.values())
    assert abs(total_norm - 1.0) < 1e-6, (
        f"Normalised weights sum to {total_norm}, expected 1.0"
    )


def test_high_accuracy_gets_higher_weight(cfg):
    """Client with higher accuracy must receive a strictly higher adaptive weight."""
    from server.adaptive_agg import AdaptiveAggregationStrategy
    from security.secure_agg import SecureAggregator

    sec = SecureAggregator(num_clients=cfg.num_hospitals, threshold=2)
    strategy = AdaptiveAggregationStrategy(config=cfg, secure_aggregator=sec)

    w_high, _ = strategy._compute_adaptive_weight("A", accuracy=0.90, num_samples=1000, max_samples=1000, round_num=1)
    w_low, _ = strategy._compute_adaptive_weight("B", accuracy=0.50, num_samples=1000, max_samples=1000, round_num=1)

    assert w_high > w_low, (
        f"High accuracy client (w={w_high:.6f}) should have higher weight than "
        f"low accuracy client (w={w_low:.6f})."
    )


def test_secure_agg_threshold_error(secure_agg):
    """SecureAggregator must raise ValueError when fewer than threshold clients submit."""
    with pytest.raises(ValueError, match="Secure aggregation requires"):
        secure_agg.aggregate_with_round({0: _make_dummy_update()}, round_num=1)
