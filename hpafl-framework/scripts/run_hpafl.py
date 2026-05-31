"""Full HPAFL training pipeline — in-process federated simulation.

Runs the complete framework without Ray subprocesses:
  - FedBN (local BN layers not aggregated)
  - DP-SGD via Opacus
  - Secure Aggregation (pairwise masking simulation)
  - Adaptive Weighted Aggregation

All clients run sequentially inside the same Python process. No Ray required.
Saves final results to results/hpafl.json.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress known-harmless library warnings that clutter training output
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="opacus")

import numpy as np
import torch

from config import (
    CLASS_NAMES, HOSPITAL_IDS, HAPFLConfig,
    KAGGLE_DATA_ROOT, KAGGLE_OUTPUT_ROOT, is_kaggle,
)
from data.dataset import create_dataloaders
from models.efficientnet_model import get_bn_layer_names, get_model
from privacy.dp_trainer import PrivacyBudgetLogger
from security.secure_agg import SecureAggregator
from server.adaptive_agg import AdaptiveAggregationStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_hpafl(config: HAPFLConfig) -> None:
    """Run the full HPAFL federated learning pipeline in a single process.

    Args:
        config: HPAFL configuration object.
    """
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for noisy in ("opacus", "opacus.accountants", "torch.utils.data.dataloader"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # faster conv for fixed input sizes
    logger.info("HPAFL simulation starting on %s", device)

    # ---- Build initial global model; fix BatchNorm → GroupNorm for DP-SGD ----
    # Opacus requires GroupNorm (DP-compatible) instead of BatchNorm.
    # We fix all models once at startup so optimizers always see GN parameters.
    from opacus.validators import ModuleValidator
    logging.getLogger("opacus").setLevel(logging.WARNING)

    def _make_dp_ready(model):
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        return model

    global_model = _make_dp_ready(get_model(config))
    bn_names = set(get_bn_layer_names(global_model))

    def get_non_bn(model):
        return [
            p.detach().cpu().numpy()
            for name, p in model.named_parameters()
            if name not in bn_names
        ]

    def set_non_bn(model, params):
        state = model.state_dict()
        non_bn = [(n, p) for n, p in model.named_parameters() if n not in bn_names]
        for (name, _), arr in zip(non_bn, params):
            state[name] = torch.tensor(arr)
        model.load_state_dict(state, strict=True)

    global_params = get_non_bn(global_model)

    # ---- Set up aggregation components ----
    secure_agg = SecureAggregator(
        num_clients=config.num_hospitals,
        threshold=max(2, config.num_hospitals - 1),
    )
    strategy = AdaptiveAggregationStrategy(
        config=config, secure_aggregator=secure_agg
    )
    budget_logger = PrivacyBudgetLogger(config.results_dir)

    # ---- Create per-hospital data loaders ----
    hospital_ids = HOSPITAL_IDS[: config.num_hospitals]
    train_loaders, val_loaders = {}, {}

    for hosp_id in hospital_ids:
        tl, vl = create_dataloaders(hosp_id, config)
        train_loaders[hosp_id] = tl
        val_loaders[hosp_id] = vl

    # FedBN: persist each hospital's local GroupNorm parameters across rounds
    # (GN layers are never sent to the server; this dict keeps them alive)
    local_gn_states: Dict[str, dict] = {h: {} for h in hospital_ids}

    # ---- Federated learning rounds ----
    from clients.local_trainer import LocalTrainer
    from evaluation.evaluate import evaluate_model

    round_metrics: List[Dict] = []
    best_f1 = 0.0
    best_params = global_params

    for rnd in range(1, config.num_rounds + 1):
        logger.info("=" * 60)
        logger.info("ROUND %d / %d", rnd, config.num_rounds)
        logger.info("=" * 60)

        lr = config.learning_rate * (0.95 ** rnd)
        fit_updates: Dict[int, List[np.ndarray]] = {}
        fit_metrics: Dict[str, Dict] = {}
        num_samples: Dict[str, int] = {}

        # ---- Local training on each hospital ----
        for i, hosp_id in enumerate(hospital_ids):
            # Create a fresh GN-fixed model every round to avoid Opacus hook
            # contamination: make_private_with_epsilon adds persistent backward
            # hooks that break ModuleValidator.validate() on the next call.
            local_model = _make_dp_ready(get_model(config)).to(device)

            # Load server's global params (non-GN layers only — FedBN)
            set_non_bn(local_model, global_params)

            # Restore this hospital's local GN parameters from the previous round
            # (FedBN: GN layers are never aggregated, must persist locally)
            if local_gn_states[hosp_id]:
                state = local_model.state_dict()
                for name, param in local_gn_states[hosp_id].items():
                    state[name] = param.to(device)
                local_model.load_state_dict(state, strict=True)

            trainer = LocalTrainer(config, hosp_id, device)
            metrics = trainer.train(
                local_model,
                train_loaders[hosp_id],
                val_loaders[hosp_id],
                learning_rate=lr,
                use_dp=True,
            )

            # Extract updated non-GN params for aggregation
            fit_updates[i] = get_non_bn(local_model)

            # Save updated local GN state for next round
            local_gn_states[hosp_id] = {
                name: param.detach().cpu()
                for name, param in local_model.named_parameters()
                if name in bn_names
            }
            fit_metrics[hosp_id] = metrics
            num_samples[hosp_id] = metrics["num_samples"]

            budget_logger.log(
                round_num=rnd,
                hospital_id=hosp_id,
                epsilon=metrics["epsilon"],
                delta=config.target_delta,
                target_epsilon=config.target_epsilon,
            )
            logger.info(
                "  Hospital %s | Loss: %.4f | Acc: %.4f | F1: %.4f | ε=%.4f",
                hosp_id,
                metrics["loss"],
                metrics["accuracy"],
                metrics["f1_macro"],
                metrics["epsilon"],
            )

        # ---- Secure Aggregation + Adaptive Weighting ----
        masked_updates = {
            i: secure_agg.mask_update(i, fit_updates[i], round_num=rnd)
            for i in range(len(hospital_ids))
        }

        max_s = max(num_samples.values())
        weights_raw = {}
        for i, hosp_id in enumerate(hospital_ids):
            w, _ = strategy._compute_adaptive_weight(
                hosp_id,
                fit_metrics[hosp_id]["accuracy"],
                num_samples[hosp_id],
                max_s,
                rnd,
            )
            weights_raw[hosp_id] = w
            strategy._update_history(hosp_id, fit_metrics[hosp_id]["accuracy"])
            strategy._participation[hosp_id] += 1

        total_w = sum(weights_raw.values())
        norm_w = {k: v / total_w for k, v in weights_raw.items()}
        strategy.save_round_weights(
            rnd,
            norm_w,
            {h: fit_metrics[h]["accuracy"] for h in hospital_ids},
            weights_raw,
        )

        # Weighted average of non-BN params
        n_layers = len(fit_updates[0])
        new_params = [np.zeros_like(fit_updates[0][k]) for k in range(n_layers)]
        for i, hosp_id in enumerate(hospital_ids):
            w = norm_w[hosp_id]
            for k in range(n_layers):
                new_params[k] += w * fit_updates[i][k]
        global_params = new_params

        # ---- Round summary ----
        round_acc = np.mean([fit_metrics[h]["accuracy"] for h in hospital_ids])
        round_f1 = np.mean([fit_metrics[h]["f1_macro"] for h in hospital_ids])
        round_eps = max(fit_metrics[h]["epsilon"] for h in hospital_ids)
        logger.info(
            "ROUND %d SUMMARY | AvgAcc: %.4f | AvgF1: %.4f | MaxEps: %.4f | Weights: %s",
            rnd,
            round_acc,
            round_f1,
            round_eps,
            {k: f"{v:.3f}" for k, v in norm_w.items()},
        )

        round_metrics.append(
            {"round": rnd, "accuracy": round_acc, "f1_macro": round_f1, "epsilon": round_eps}
        )
        if round_f1 > best_f1:
            best_f1 = round_f1
            best_params = [p.copy() for p in global_params]

    # ---- Save final model checkpoint ----
    final_model = _make_dp_ready(get_model(config))
    set_non_bn(final_model, best_params)
    ckpt = Path(config.checkpoint_dir) / "global_model_final.pt"
    torch.save(final_model.state_dict(), ckpt)
    logger.info("Global model checkpoint saved → %s", ckpt)

    # ---- Final evaluation on all hospital val sets ----
    final_model.to(device).eval()

    from torch.utils.data import ConcatDataset
    all_val = ConcatDataset(list(val_loaders[h].dataset for h in hospital_ids))
    from torch.utils.data import DataLoader as TorchDataLoader
    combined_loader = TorchDataLoader(all_val, batch_size=config.batch_size, shuffle=False)

    final_metrics = evaluate_model(final_model, combined_loader, CLASS_NAMES, device)
    max_epsilon = max(rnd_m["epsilon"] for rnd_m in round_metrics) if round_metrics else 0.0

    results = {
        "system": "hpafl",
        "accuracy": final_metrics["accuracy"],
        "f1_macro": final_metrics["f1_macro"],
        "roc_auc_ovr": final_metrics["roc_auc_ovr"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_per_class": final_metrics["f1_per_class"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "epsilon": max_epsilon,
        "delta": config.target_delta,
        "num_rounds_trained": config.num_rounds,
        "num_hospitals": config.num_hospitals,
        "components": ["FedBN", "DP-SGD", "SecureAgg", "AdaptiveWeighting"],
        "round_history": round_metrics,
    }

    out_path = Path(config.results_dir) / "hpafl.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("HPAFL results saved → %s", out_path)
    logger.info(
        "FINAL | Acc=%.4f | F1=%.4f | AUC=%.4f | ε=%.4f",
        results["accuracy"],
        results["f1_macro"],
        results["roc_auc_ovr"],
        max_epsilon,
    )


if __name__ == "__main__":
    cfg = HAPFLConfig()
    if is_kaggle():
        cfg.data_root = KAGGLE_DATA_ROOT
        cfg.output_root = KAGGLE_OUTPUT_ROOT
    else:
        _root = Path(__file__).resolve().parents[1]
        cfg.data_root = str(_root.parent / "archive")
        cfg.output_root = str(_root)
    run_hpafl(cfg)
