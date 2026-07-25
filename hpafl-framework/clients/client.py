"""Flower NumPyClient implementation for HPAFL hospital nodes.

Implements the FedBN strategy: BatchNorm parameters are excluded from
get_parameters() and set_parameters() so they remain local to each hospital.
"""

import logging
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from clients.local_trainer import LocalTrainer
from config import CLASS_NAMES, HAPFLConfig
from data.dataset import create_dataloaders
from models.efficientnet_model import get_bn_layer_names, get_model

logger = logging.getLogger(__name__)


class HAM10000FlowerClient(fl.client.NumPyClient):
    """Flower client for one hospital node implementing FedBN.

    Key FedBN behaviour:
        - get_parameters() EXCLUDES BN layer params (they stay local).
        - set_parameters() updates ONLY non-BN layers; BN layers unchanged.
        - BN layers are trained locally and never shared with the server.

    Args:
        hospital_id: Hospital identifier ('A', 'B', or 'C').
        config: HPAFL configuration object.
    """

    def __init__(self, hospital_id: str, config: HAPFLConfig) -> None:
        self.hospital_id = hospital_id
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = get_model(config).to(self.device)
        self.bn_layer_names = set(get_bn_layer_names(self.model))

        self.train_loader, self.val_loader = create_dataloaders(hospital_id, config)
        self.trainer = LocalTrainer(config, hospital_id, self.device)
        self._round: int = 0

        logger.info(
            "Hospital %s client initialised on %s. "
            "BN params excluded from federation: %d",
            hospital_id,
            self.device,
            len(self.bn_layer_names),
        )

    def _non_bn_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """Return a list of (name, param) tuples for non-BN parameters.

        Returns:
            Named parameters excluding those belonging to BatchNorm layers.
        """
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if name not in self.bn_layer_names
        ]

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return non-BN model parameters as numpy arrays.

        BN parameters are intentionally excluded (FedBN strategy).

        Args:
            config: Flower-provided configuration dict (unused here).

        Returns:
            List of numpy arrays representing non-BN parameter values.
        """
        self.model.eval()
        return [
            param.detach().cpu().numpy()
            for _, param in self._non_bn_params()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set non-BN model parameters from numpy arrays.

        BN layers remain unchanged (FedBN: local BN is preserved).

        Args:
            parameters: List of numpy arrays (same order as get_parameters).

        Raises:
            ValueError: If parameter count or shapes don't match model structure.
            TypeError: If parameters are not numpy arrays.
        """
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(
                f"parameters must be a list/tuple of numpy arrays; got {type(parameters)}"
            )

        non_bn = self._non_bn_params()
        if len(parameters) != len(non_bn):
            raise ValueError(
                f"Parameter count mismatch: expected {len(non_bn)}, "
                f"got {len(parameters)}. This usually means the global model "
                f"architecture changed or there's a version mismatch."
            )

        # Validate shapes before loading
        for i, ((name, param), arr) in enumerate(zip(non_bn, parameters)):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"Parameter {i} ({name}) is not a numpy array; got {type(arr)}"
                )
            expected_shape = tuple(param.shape)
            actual_shape = tuple(arr.shape)
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Shape mismatch for parameter {i} ({name}): "
                    f"expected {expected_shape}, got {actual_shape}"
                )

        state_dict = self.model.state_dict()
        for (name, _), arr in zip(non_bn, parameters):
            state_dict[name] = torch.tensor(arr, device=self.device)

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load state dict for Hospital {self.hospital_id}: {str(e)}. "
                f"This may indicate BatchNorm layers were incorrectly excluded."
            ) from e

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training round.

        Args:
            parameters: Global non-BN parameters from the server.
            config: Per-round config from on_fit_config_fn (round, lr, etc.).

        Returns:
            Tuple of (updated_params, num_examples, metrics_dict).
            metrics_dict keys: loss, accuracy, epsilon, hospital_id, round.
        """
        self._round = config.get("round", self._round + 1)
        lr = float(config.get("learning_rate", self.config.learning_rate))
        use_dp = config.get("use_dp", True)

        self.set_parameters(parameters)

        metrics = self.trainer.train(
            self.model,
            self.train_loader,
            self.val_loader,
            learning_rate=lr,
            use_dp=use_dp,
        )

        updated_params = self.get_parameters({})
        num_examples = metrics["num_samples"]

        fit_metrics = {
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "epsilon": float(metrics["epsilon"]),
            "hospital_id": self.hospital_id,
            "round": self._round,
        }

        logger.info(
            "Hospital %s | Round %d | Loss: %.4f | Acc: %.4f | F1: %.4f | ε=%.4f",
            self.hospital_id,
            self._round,
            metrics["loss"],
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["epsilon"],
        )
        return updated_params, num_examples, fit_metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """Local evaluation round.

        Args:
            parameters: Global non-BN parameters from the server.
            config: Per-round config from on_evaluate_config_fn.

        Returns:
            Tuple of (loss, num_examples, metrics_dict).
            metrics_dict keys: accuracy, f1_macro, roc_auc, per_class_f1, hospital_id.
        """
        self.set_parameters(parameters)
        self.model.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        from models.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=self.config.focal_gamma)

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels_dev = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                loss = criterion(logits, labels_dev)
                total_loss += loss.item()
                n_batches += 1

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.tolist())

        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)
        all_probs_arr = np.array(all_probs)

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

        try:
            roc_auc = float(
                roc_auc_score(
                    all_labels_arr,
                    all_probs_arr,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            roc_auc = 0.0

        avg_loss = total_loss / max(n_batches, 1)
        num_examples = len(self.val_loader.dataset)

        eval_metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "per_class_f1": {
                CLASS_NAMES[i]: round(per_class_f1[i], 4)
                for i in range(len(CLASS_NAMES))
            },
            "hospital_id": self.hospital_id,
        }
        logger.info(
            "Hospital %s EVAL | Acc: %.4f | F1: %.4f | AUC: %.4f",
            self.hospital_id,
            accuracy,
            f1_macro,
            roc_auc,
        )
        return avg_loss, num_examples, eval_metrics


def start_client(
    hospital_id: str,
    server_address: str,
    config: HAPFLConfig,
) -> None:
    """Start a Flower client for the given hospital.

    Args:
        hospital_id: Hospital identifier ('A', 'B', or 'C').
        server_address: Flower server gRPC address (e.g. '127.0.0.1:8080').
        config: HPAFL configuration object.
    """
    client = HAM10000FlowerClient(hospital_id, config)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
