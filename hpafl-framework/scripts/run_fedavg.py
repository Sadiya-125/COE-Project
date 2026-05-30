"""Standard FedAvg baseline — in-process federated simulation.

No DP, no secure aggregation, no adaptive weights. Standard sample-weighted
FedAvg as described in McMahan et al. (AISTATS 2017).
All clients run sequentially in the same Python process.
Saves results to results/fedavg.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CLASS_NAMES, HOSPITAL_IDS, HAPFLConfig
from data.dataset import create_dataloaders
from evaluation.evaluate import evaluate_model
from models.efficientnet_model import get_model
from models.focal_loss import FocalLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_fedavg(config: HAPFLConfig) -> None:
    """Run standard FedAvg in-process and save results.

    Args:
        config: HPAFL configuration object.
    """
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("FedAvg simulation starting on %s", device)

    hospital_ids = HOSPITAL_IDS[: config.num_hospitals]

    # Build shared parameter store (all params, including BN — vanilla FedAvg)
    global_model = get_model(config)

    def get_all_params(model):
        return [p.detach().cpu().numpy() for p in model.parameters()]

    def set_all_params(model, params):
        state = model.state_dict()
        for (name, _), arr in zip(model.named_parameters(), params):
            state[name] = torch.tensor(arr)
        model.load_state_dict(state, strict=True)

    global_params = get_all_params(global_model)

    # Per-hospital data loaders and local models
    train_loaders, val_loaders, local_models = {}, {}, {}
    for hosp_id in hospital_ids:
        tl, vl = create_dataloaders(hosp_id, config)
        train_loaders[hosp_id] = tl
        val_loaders[hosp_id] = vl
        local_models[hosp_id] = get_model(config).to(device)

    round_metrics = []
    best_f1 = 0.0
    best_params = global_params

    for rnd in range(1, config.num_rounds + 1):
        logger.info("FedAvg ROUND %d / %d", rnd, config.num_rounds)
        lr = config.learning_rate * (0.95 ** rnd)
        local_updates: Dict[str, List[np.ndarray]] = {}
        n_samples: Dict[str, int] = {}

        # Local training (no DP)
        for hosp_id in hospital_ids:
            set_all_params(local_models[hosp_id], global_params)
            model = local_models[hosp_id]
            criterion = FocalLoss(gamma=config.focal_gamma)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

            model.train()
            for _ in range(config.local_epochs):
                for images, labels in train_loaders[hosp_id]:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

            local_updates[hosp_id] = get_all_params(model)
            n_samples[hosp_id] = len(train_loaders[hosp_id].dataset)

        # Sample-weighted FedAvg aggregation
        total_n = sum(n_samples.values())
        n_layers = len(global_params)
        new_params = [np.zeros_like(global_params[k]) for k in range(n_layers)]
        for hosp_id in hospital_ids:
            w = n_samples[hosp_id] / total_n
            for k in range(n_layers):
                new_params[k] += w * local_updates[hosp_id][k]
        global_params = new_params

        # Quick validation on first hospital
        set_all_params(local_models[hospital_ids[0]], global_params)
        val_metrics = evaluate_model(
            local_models[hospital_ids[0]],
            val_loaders[hospital_ids[0]],
            CLASS_NAMES,
            device,
        )
        logger.info(
            "  Round %d | Acc: %.4f | F1: %.4f",
            rnd, val_metrics["accuracy"], val_metrics["f1_macro"],
        )
        round_metrics.append({
            "round": rnd,
            "accuracy": val_metrics["accuracy"],
            "f1_macro": val_metrics["f1_macro"],
        })
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_params = [p.copy() for p in global_params]

    # Final evaluation
    final_model = get_model(config).to(device)
    set_all_params(final_model, best_params)
    from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader
    all_val = ConcatDataset([val_loaders[h].dataset for h in hospital_ids])
    combined_loader = TorchDataLoader(all_val, batch_size=config.batch_size, shuffle=False)
    final_metrics = evaluate_model(final_model, combined_loader, CLASS_NAMES, device)

    results = {
        "system": "fedavg",
        "accuracy": final_metrics["accuracy"],
        "f1_macro": final_metrics["f1_macro"],
        "roc_auc_ovr": final_metrics["roc_auc_ovr"],
        "precision_macro": final_metrics["precision_macro"],
        "recall_macro": final_metrics["recall_macro"],
        "f1_per_class": final_metrics["f1_per_class"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "epsilon": float("inf"),
        "num_rounds_trained": config.num_rounds,
        "round_history": round_metrics,
    }

    out_path = Path(config.results_dir) / "fedavg.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("FedAvg results saved → %s", out_path)
    logger.info(
        "FINAL | Acc=%.4f | F1=%.4f | AUC=%.4f",
        results["accuracy"], results["f1_macro"], results["roc_auc_ovr"],
    )


if __name__ == "__main__":
    cfg = HAPFLConfig()
    cfg.data_root = str(Path(__file__).resolve().parents[2] / "archive")
    run_fedavg(cfg)
