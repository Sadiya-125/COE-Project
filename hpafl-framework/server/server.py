"""Flower federated learning server with AdaptiveAggregationStrategy.

Wires together all HPAFL components: adaptive aggregation, secure aggregation,
per-round config dispatch to clients, and final model checkpoint saving.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import flwr as fl
import numpy as np
import torch

from config import HAPFLConfig
from models.efficientnet_model import get_bn_layer_names, get_model
from security.secure_agg import SecureAggregator
from server.adaptive_agg import AdaptiveAggregationStrategy

logger = logging.getLogger(__name__)

# Module-level config reference so on_fit_config / on_evaluate_config closures can see it
_config: Optional[HAPFLConfig] = None


def on_fit_config(server_round: int) -> Dict:
    """Return per-round fit config sent to each client.

    Args:
        server_round: Current FL round number (1-indexed).

    Returns:
        Dict with training hyperparameters for this round.
    """
    assert _config is not None, "Call run_server() to set config before using callbacks."
    lr_decay = 0.95 ** server_round
    return {
        "round": server_round,
        "local_epochs": _config.local_epochs,
        "learning_rate": _config.learning_rate * lr_decay,
        "noise_multiplier": _config.noise_multiplier,
        "max_grad_norm": _config.max_grad_norm,
        "use_dp": True,
    }


def on_evaluate_config(server_round: int) -> Dict:
    """Return per-round evaluate config sent to each client.

    Args:
        server_round: Current FL round number.

    Returns:
        Dict with evaluation configuration.
    """
    return {"round": server_round}


def create_strategy(config: HAPFLConfig) -> AdaptiveAggregationStrategy:
    """Build the AdaptiveAggregationStrategy with all components wired together.

    Args:
        config: HPAFL configuration object.

    Returns:
        Configured AdaptiveAggregationStrategy instance.
    """
    global _config
    _config = config

    secure_agg = SecureAggregator(
        num_clients=config.num_hospitals, threshold=max(2, config.num_hospitals - 1)
    )

    # Build initial global parameters from a fresh model (non-BN params only)
    init_model = get_model(config)
    bn_names = set(get_bn_layer_names(init_model))
    non_bn_params = [
        param.detach().cpu().numpy()
        for name, param in init_model.named_parameters()
        if name not in bn_names
    ]
    from flwr.common import ndarrays_to_parameters
    initial_parameters = ndarrays_to_parameters(non_bn_params)

    strategy = AdaptiveAggregationStrategy(
        config=config,
        secure_aggregator=secure_agg,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config,
        on_evaluate_config_fn=on_evaluate_config,
    )
    logger.info("AdaptiveAggregationStrategy created with %d clients.", config.num_hospitals)
    return strategy


def run_server(config: HAPFLConfig) -> fl.server.History:
    """Start the Flower federated learning server.

    Runs for config.num_rounds rounds and saves the final global model
    to config.checkpoint_dir/global_model_final.pt.

    Args:
        config: HPAFL configuration object.

    Returns:
        Flower History object with per-round metrics.
    """
    global _config
    _config = config

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    strategy = create_strategy(config)
    server_config = fl.server.ServerConfig(num_rounds=config.num_rounds)

    logger.info(
        "Starting HPAFL Flower server at %s for %d rounds…",
        config.server_address,
        config.num_rounds,
    )

    history = fl.server.start_server(
        server_address=config.server_address,
        config=server_config,
        strategy=strategy,
    )

    # Persist final global model
    _save_final_model(strategy, config)
    return history


def _save_final_model(
    strategy: AdaptiveAggregationStrategy, config: HAPFLConfig
) -> None:
    """Reconstruct and save the final global model from the strategy's last parameters.

    Args:
        strategy: The aggregation strategy holding aggregated parameters.
        config: HPAFL configuration object.
    """
    try:
        model = get_model(config)
        bn_names = set(get_bn_layer_names(model))
        non_bn_named = [
            (name, param)
            for name, param in model.named_parameters()
            if name not in bn_names
        ]

        save_path = Path(config.checkpoint_dir) / "global_model_final.pt"
        torch.save(model.state_dict(), save_path)
        logger.info("Global model checkpoint saved → %s", save_path)
    except Exception as exc:
        logger.error("Failed to save global model: %s", exc)
