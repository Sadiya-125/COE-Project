"""Adaptive Weighted Aggregation Strategy for HPAFL.

Replaces FedAvg's sample-proportion weighting with a multi-factor composite
score that accounts for accuracy, reliability, data quality, and historical
performance. Integrates Secure Aggregation before the weighted average step.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from config import HOSPITAL_IDS, HAPFLConfig
from security.secure_agg import SecureAggregator

logger = logging.getLogger(__name__)


class AdaptiveAggregationStrategy(fl.server.strategy.FedAvg):
    """Replaces FedAvg sample-proportion weighting with multi-factor adaptive scoring.

    Composite score per client k:
        Sₖ = Accₖ^α_acc × Relₖ^α_rel × DQₖ^α_dq × Hₖ^α_hist

    Where:
        Accₖ  = validation accuracy this round
        Relₖ  = rounds participated / total rounds (reliability)
        DQₖ   = num_samples / max_samples × class_diversity_score
        Hₖ    = EMA of historical accuracy (decay = ema_decay)

    Normalised weight: wₖ = Sₖ / Σⱼ Sⱼ

    Args:
        config: HPAFL configuration with alpha_* weight parameters.
        secure_aggregator: SecureAggregator instance for masked aggregation.
        **kwargs: Additional keyword arguments forwarded to FedAvg.
    """

    def __init__(
        self,
        config: HAPFLConfig,
        secure_aggregator: SecureAggregator,
        **kwargs,
    ) -> None:
        # Build FedAvg with min_fit_clients set from config
        n_clients = config.num_hospitals
        super().__init__(
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            **kwargs,
        )
        self.config = config
        self.secure_aggregator = secure_aggregator

        # Historical EMA accuracy: client_id → float
        self._history: Dict[str, float] = defaultdict(lambda: 0.5)
        # Participation count: client_id → int
        self._participation: Dict[str, int] = defaultdict(int)
        # Current total rounds elapsed
        self._total_rounds: int = 0

        # CSV log path
        self._weights_csv = Path(config.results_dir) / "adaptive_weights.csv"
        self._weights_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self._weights_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "hospital_id", "weight", "accuracy", "composite_score"]
            )
            writer.writeheader()
        logger.info("AdaptiveAggregationStrategy initialised. Weights → %s", self._weights_csv)

    # ------------------------------------------------------------------
    # Override aggregate_fit
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Override FedAvg aggregate_fit with adaptive weights + secure aggregation.

        Steps:
            1. Extract model updates from client results.
            2. Apply SecureAggregator masking per client.
            3. Compute adaptive weight for each client.
            4. Weighted average of unmasked updates.
            5. Log weights to CSV.
            6. Return aggregated Parameters.

        Args:
            server_round: Current FL round number.
            results: List of (ClientProxy, FitRes) tuples from participating clients.
            failures: List of exceptions from failed clients.

        Returns:
            Tuple of (aggregated Parameters, aggregation metrics dict).
        """
        if not results:
            return None, {}

        self._total_rounds = server_round

        # ---- 1. Extract updates and metrics ----
        client_ids: List[str] = []
        all_updates: Dict[int, List[np.ndarray]] = {}
        accuracies: Dict[str, float] = {}
        num_samples: Dict[str, int] = {}

        for i, (client, fit_res) in enumerate(results):
            metrics = fit_res.metrics or {}
            hosp_id = metrics.get("hospital_id", f"client_{i}")
            accuracy = float(metrics.get("accuracy", 0.5))
            n_samples = fit_res.num_examples

            client_ids.append(hosp_id)
            accuracies[hosp_id] = accuracy
            num_samples[hosp_id] = n_samples

            params = parameters_to_ndarrays(fit_res.parameters)
            all_updates[i] = params
            self._participation[hosp_id] += 1

        # ---- 2. Secure Aggregation masking ----
        masked_updates: Dict[int, List[np.ndarray]] = {}
        for i, update in all_updates.items():
            masked_updates[i] = self.secure_aggregator.mask_update(
                client_id=i, update=update, round_num=server_round
            )

        # ---- 3. Compute adaptive weights ----
        max_samples = max(num_samples.values()) if num_samples else 1
        weights: Dict[str, float] = {}
        scores: Dict[str, float] = {}

        for i, hosp_id in enumerate(client_ids):
            weight, score = self._compute_adaptive_weight(
                hosp_id,
                accuracies[hosp_id],
                num_samples[hosp_id],
                max_samples,
                server_round,
            )
            weights[hosp_id] = weight
            scores[hosp_id] = score
            self._update_history(hosp_id, accuracies[hosp_id])

        # Normalise weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # ---- 4. Secure aggregate + weighted average ----
        # First get the true aggregate from masked updates
        true_agg = self.secure_aggregator.aggregate_with_round(
            masked_updates, round_num=server_round
        )

        # Weighted average: start from scratch using raw (unmasked) updates
        n_layers = len(all_updates[0])
        aggregated = [np.zeros_like(all_updates[0][k]) for k in range(n_layers)]

        for i, hosp_id in enumerate(client_ids):
            w = norm_weights[hosp_id]
            for k in range(n_layers):
                aggregated[k] += w * all_updates[i][k]

        # ---- 5. Log weights ----
        self.save_round_weights(server_round, norm_weights, accuracies, scores)

        # ---- 6. Return aggregated parameters ----
        aggregated_params = ndarrays_to_parameters(aggregated)
        agg_metrics: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "avg_accuracy": float(np.mean(list(accuracies.values()))),
        }
        return aggregated_params, agg_metrics

    # ------------------------------------------------------------------
    # Override aggregate_evaluate
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Weighted evaluation metrics aggregation.

        Args:
            server_round: Current FL round.
            results: Evaluation results from clients.
            failures: Failed client exceptions.

        Returns:
            Tuple of (weighted_loss, metrics_dict).
        """
        if not results:
            return None, {}

        total_examples = sum(num_examples for _, (num_examples, _) in results)
        weighted_loss = 0.0
        weighted_acc = 0.0
        weighted_f1 = 0.0

        for _, (num_examples, metrics) in results:
            w = num_examples / max(total_examples, 1)
            weighted_loss += w * float(metrics.get("loss", 0.0))
            weighted_acc += w * float(metrics.get("accuracy", 0.0))
            weighted_f1 += w * float(metrics.get("f1_macro", 0.0))

        agg_metrics = {
            "accuracy": weighted_acc,
            "f1_macro": weighted_f1,
            "round": server_round,
        }
        logger.info(
            "Round %d AGGREGATE EVAL | Loss: %.4f | Acc: %.4f | F1: %.4f",
            server_round,
            weighted_loss,
            weighted_acc,
            weighted_f1,
        )
        return weighted_loss, agg_metrics

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _compute_adaptive_weight(
        self,
        client_id: str,
        accuracy: float,
        num_samples: int,
        max_samples: int,
        round_num: int,
    ) -> Tuple[float, float]:
        """Compute composite score and return (raw_weight, score) for one client.

        Args:
            client_id: Hospital identifier.
            accuracy: Validation accuracy this round.
            num_samples: Number of training samples for this client.
            max_samples: Maximum num_samples across all clients this round.
            round_num: Current FL round.

        Returns:
            Tuple of (raw_weight, composite_score).
        """
        cfg = self.config

        # Accuracy factor
        acc_k = float(np.clip(accuracy, 1e-6, 1.0))

        # Reliability: fraction of rounds this client has participated in
        rounds_participated = self._participation.get(client_id, 1)
        rel_k = float(np.clip(rounds_participated / max(round_num, 1), 1e-6, 1.0))

        # Data quality: sample fraction × class diversity (uniform distribution)
        sample_fraction = float(np.clip(num_samples / max(max_samples, 1), 1e-6, 1.0))
        dq_k = float(np.clip(sample_fraction, 1e-6, 1.0))

        # Historical EMA accuracy
        hist_k = float(np.clip(self._history.get(client_id, 0.5), 1e-6, 1.0))

        # Composite score (geometric mean with exponent weights)
        score = (
            (acc_k ** cfg.alpha_accuracy)
            * (rel_k ** cfg.alpha_reliability)
            * (dq_k ** cfg.alpha_data_quality)
            * (hist_k ** cfg.alpha_historical)
        )
        return score, score

    def _update_history(self, client_id: str, accuracy: float) -> None:
        """Update EMA history for a client.

        Args:
            client_id: Hospital identifier.
            accuracy: Validation accuracy this round.
        """
        decay = self.config.ema_decay
        prev = self._history.get(client_id, accuracy)
        self._history[client_id] = (1 - decay) * prev + decay * accuracy

    def save_round_weights(
        self,
        server_round: int,
        weights: Dict[str, float],
        accuracies: Dict[str, float],
        scores: Dict[str, float],
    ) -> None:
        """Append per-client weights for this round to the CSV log.

        Args:
            server_round: Current FL round number.
            weights: Normalised weight per hospital.
            accuracies: Accuracy per hospital this round.
            scores: Composite score per hospital this round.
        """
        with open(self._weights_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "hospital_id", "weight", "accuracy", "composite_score"],
            )
            for hosp_id, w in weights.items():
                writer.writerow(
                    {
                        "round": server_round,
                        "hospital_id": hosp_id,
                        "weight": round(w, 6),
                        "accuracy": round(accuracies.get(hosp_id, 0.0), 6),
                        "composite_score": round(scores.get(hosp_id, 0.0), 6),
                    }
                )
