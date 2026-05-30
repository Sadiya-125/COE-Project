"""Model evaluation, baseline comparison, and Plotly visualisation for HPAFL.

Generates interactive HTML charts for learning curves, privacy budgets,
adaptive weights, and confusion matrices, plus a formatted comparison table.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config import CLASS_NAMES, CLASS_FULL_NAMES

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    """Full evaluation returning accuracy, precision, recall, f1_macro,
    f1_per_class, roc_auc_ovr, and confusion_matrix.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the evaluation dataset.
        class_names: List of class name strings.
        device: Device for inference.

    Returns:
        Dict with keys: accuracy, precision_macro, recall_macro, f1_macro,
        f1_per_class, roc_auc_ovr, confusion_matrix (list of lists).
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    accuracy = float(accuracy_score(labels_arr, preds_arr))
    precision_macro = float(
        precision_score(labels_arr, preds_arr, average="macro", zero_division=0)
    )
    recall_macro = float(
        recall_score(labels_arr, preds_arr, average="macro", zero_division=0)
    )
    f1_macro = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))
    f1_per_class = f1_score(
        labels_arr, preds_arr, average=None, zero_division=0,
        labels=list(range(len(class_names)))
    ).tolist()

    try:
        roc_auc = float(
            roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
        )
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(len(class_names)))).tolist()

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_per_class": {class_names[i]: round(f1_per_class[i], 4) for i in range(len(class_names))},
        "roc_auc_ovr": roc_auc,
        "confusion_matrix": cm,
        "num_samples": len(labels_arr),
    }


def compare_baselines(results_dir: str) -> pd.DataFrame:
    """Load results JSON files for all baselines and return a comparison DataFrame.

    Looks for: results/centralized.json, results/fedavg.json,
    results/fedprox_dp.json, results/hpafl.json.

    Args:
        results_dir: Path to the results directory.

    Returns:
        DataFrame with one row per system and metric columns.
    """
    systems = {
        "Centralized": "centralized.json",
        "FedAvg": "fedavg.json",
        "FedProx+DP": "fedprox_dp.json",
        "HPAFL (ours)": "hpafl.json",
    }
    results_path = Path(results_dir)
    rows = []
    for system_name, fname in systems.items():
        fpath = results_path / fname
        if not fpath.exists():
            logger.warning("Results file not found: %s", fpath)
            continue
        with open(fpath) as f:
            data = json.load(f)
        row = {
            "System": system_name,
            "Accuracy": round(data.get("accuracy", 0.0), 4),
            "F1-Macro": round(data.get("f1_macro", 0.0), 4),
            "ROC-AUC": round(data.get("roc_auc_ovr", 0.0), 4),
            "Precision": round(data.get("precision_macro", 0.0), 4),
            "Recall": round(data.get("recall_macro", 0.0), 4),
            "Epsilon (ε)": round(data.get("epsilon", float("inf")), 4),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("System")


def plot_learning_curves(results_dir: str) -> None:
    """Plotly line chart of accuracy per hospital per FL round.

    Reads adaptive_weights.csv for per-hospital accuracy data.
    Saves to results/plots/learning_curves.html.

    Args:
        results_dir: Path to the results directory.
    """
    weights_csv = Path(results_dir) / "adaptive_weights.csv"
    if not weights_csv.exists():
        logger.warning("adaptive_weights.csv not found; skipping learning curves.")
        return

    df = pd.read_csv(weights_csv)
    if df.empty:
        return

    fig = px.line(
        df,
        x="round",
        y="accuracy",
        color="hospital_id",
        title="Federated Learning: Accuracy per Hospital per Round",
        labels={"round": "FL Round", "accuracy": "Validation Accuracy", "hospital_id": "Hospital"},
        markers=True,
    )
    fig.update_layout(template="plotly_white", font_size=13)

    out_path = Path(results_dir) / "plots" / "learning_curves.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    logger.info("Learning curves saved → %s", out_path)


def plot_privacy_budget(results_dir: str) -> None:
    """Plotly line chart of cumulative epsilon per hospital per FL round.

    Reads privacy_budget.csv.
    Saves to results/plots/privacy_budget.html.

    Args:
        results_dir: Path to the results directory.
    """
    budget_csv = Path(results_dir) / "privacy_budget.csv"
    if not budget_csv.exists():
        logger.warning("privacy_budget.csv not found; skipping privacy budget plot.")
        return

    df = pd.read_csv(budget_csv)
    if df.empty:
        return

    fig = px.line(
        df,
        x="round",
        y="epsilon",
        color="hospital_id",
        title="Privacy Budget Consumption (Cumulative ε) per Hospital",
        labels={"round": "FL Round", "epsilon": "Cumulative ε", "hospital_id": "Hospital"},
        markers=True,
    )
    fig.add_hline(y=8.0, line_dash="dash", line_color="red", annotation_text="ε budget = 8.0")
    fig.update_layout(template="plotly_white", font_size=13)

    out_path = Path(results_dir) / "plots" / "privacy_budget.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    logger.info("Privacy budget plot saved → %s", out_path)


def plot_adaptive_weights(results_dir: str) -> None:
    """Plotly stacked bar of per-client adaptive weights per FL round.

    Reads adaptive_weights.csv.
    Saves to results/plots/adaptive_weights.html.

    Args:
        results_dir: Path to the results directory.
    """
    weights_csv = Path(results_dir) / "adaptive_weights.csv"
    if not weights_csv.exists():
        logger.warning("adaptive_weights.csv not found; skipping adaptive weights plot.")
        return

    df = pd.read_csv(weights_csv)
    if df.empty:
        return

    fig = px.bar(
        df,
        x="round",
        y="weight",
        color="hospital_id",
        barmode="stack",
        title="Adaptive Aggregation Weight Evolution per Round",
        labels={"round": "FL Round", "weight": "Normalised Weight", "hospital_id": "Hospital"},
    )
    fig.update_layout(template="plotly_white", font_size=13)

    out_path = Path(results_dir) / "plots" / "adaptive_weights.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    logger.info("Adaptive weights plot saved → %s", out_path)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], title: str
) -> None:
    """Plotly heatmap confusion matrix with class names.

    Saves to results/plots/confusion_matrix.html.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        class_names: List of class name strings.
        title: Chart title.
    """
    cm_arr = np.array(cm)
    # Normalise to percentages row-wise
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=[[f"{cm_arr[r][c]}" for c in range(len(class_names))]
                  for r in range(len(class_names))],
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        font_size=12,
        template="plotly_white",
    )

    out_path = Path("./results/plots/confusion_matrix.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    logger.info("Confusion matrix saved → %s", out_path)


def generate_full_report(results_dir: str) -> None:
    """Run all evaluation functions and print a formatted comparison table.

    Args:
        results_dir: Path to the results directory.
    """
    print("\n" + "=" * 70)
    print("HPAFL SYSTEM COMPARISON REPORT")
    print("=" * 70)

    df = compare_baselines(results_dir)
    if not df.empty:
        print(df.to_string())
    else:
        print("No results files found. Run the training scripts first.")

    print("\nGenerating visualisation plots…")
    plot_learning_curves(results_dir)
    plot_privacy_budget(results_dir)
    plot_adaptive_weights(results_dir)

    hpafl_path = Path(results_dir) / "hpafl.json"
    if hpafl_path.exists():
        with open(hpafl_path) as f:
            data = json.load(f)
        cm = data.get("confusion_matrix")
        if cm:
            plot_confusion_matrix(
                np.array(cm), CLASS_NAMES, "HPAFL Confusion Matrix"
            )

    print("\nAll plots saved to", Path(results_dir) / "plots")
    print("=" * 70 + "\n")
