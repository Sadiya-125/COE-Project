"""Central configuration module for the HPAFL framework.

All hyperparameters are defined here. No magic numbers elsewhere in the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle notebook/script environment."""
    return os.path.exists("/kaggle/input")


@dataclass
class HAPFLConfig:
    """Master configuration for the Hybrid Privacy-Aware Federated Learning framework.

    On Kaggle set only two fields and everything else resolves automatically::

        cfg = HAPFLConfig()
        cfg.data_root   = "/kaggle/input/skin-cancer-mnist-ham10000"
        cfg.output_root = "/kaggle/working"

    Args:
        data_root: Read-only path to HAM10000 images and metadata CSV.
            Kaggle: /kaggle/input/skin-cancer-mnist-ham10000
            Local:  ../archive  (relative to hpafl-framework/)
        output_root: Writable root for all generated files (partitions,
            checkpoints, results).  Kaggle: /kaggle/working  Local: .
        num_classes: Number of skin lesion classes.
        image_size: Input image size (square).
        num_hospitals: Number of federated hospital clients.
        hospital_splits: Fraction of data assigned to each hospital.
        dirichlet_alpha: Concentration parameter for non-IID Dirichlet split.
        model_name: EfficientNet variant to use.
        pretrained: Whether to load ImageNet pretrained weights.
        dropout_rate: Dropout probability in classifier head.
        hidden_dim: Hidden dimension of classifier head.
        local_epochs: Local training epochs per FL round.
        batch_size: Mini-batch size (64 on T4 16 GB; try 128 if headroom allows).
        num_workers: DataLoader workers (4 on Kaggle T4).
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        focal_gamma: Focusing parameter for FocalLoss (0 = CrossEntropy).
        focal_alpha: Class weighting factor for severe imbalance (2.0-5.0).
        use_augmentation: Enable data augmentation for improved generalization.
        augmentation_strength: Augmentation intensity ("low", "medium", "high").
        num_rounds: Total federated learning communication rounds.
        fraction_fit: Fraction of clients selected per round for training.
        fraction_evaluate: Fraction of clients selected per round for evaluation.
        server_address: gRPC address for Flower server.
        noise_multiplier: Gaussian noise multiplier for DP-SGD.
        max_grad_norm: Per-sample gradient clipping threshold.
        target_epsilon: Privacy budget upper bound (epsilon).
        target_delta: Privacy failure probability (delta).
        alpha_accuracy: Weight of accuracy factor in adaptive composite score.
        alpha_reliability: Weight of reliability factor in adaptive composite score.
        alpha_data_quality: Weight of data quality factor in adaptive composite score.
        alpha_historical: Weight of historical accuracy EMA in adaptive composite score.
        ema_decay: Exponential moving average decay for historical accuracy.
        api_host: FastAPI server host.
        api_port: FastAPI server port.
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    # Read-only dataset location
    data_root: str = "./data/ham10000"
    # Writable output root — partitions, checkpoints, results all go here
    output_root: str = "."

    # ── Data ─────────────────────────────────────────────────────────────────
    num_classes: int = 7
    image_size: int = 224
    num_hospitals: int = 3
    hospital_splits: tuple = (0.35, 0.35, 0.30)
    dirichlet_alpha: float = 0.5

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout_rate: float = 0.3
    hidden_dim: int = 512

    # ── Training ──────────────────────────────────────────────────────────────
    local_epochs: int = 5
    batch_size: int = 64        # comfortable on T4 16 GB; try 128 if headroom allows
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    focal_gamma: float = 2.0    # Focusing parameter for FocalLoss
    focal_alpha: float = 2.0    # Class weighting for severe imbalance (try 3.0-5.0)
    use_augmentation: bool = True  # Enable data augmentation
    augmentation_strength: str = "high"  # Options: "low", "medium", "high"
    num_workers: int = 4        # Kaggle T4 ships with 4 CPU cores

    # ── Federated Learning ────────────────────────────────────────────────────
    num_rounds: int = 20
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    server_address: str = "0.0.0.0:8080"

    # ── Differential Privacy ──────────────────────────────────────────────────
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    target_epsilon: float = 8.0
    target_delta: float = 1e-5

    # ── Adaptive Aggregation weights (must sum to 1.0) ────────────────────────
    alpha_accuracy: float = 0.40
    alpha_reliability: float = 0.30
    alpha_data_quality: float = 0.20
    alpha_historical: float = 0.10
    ema_decay: float = 0.30

    # ── Deployment ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Derived paths (computed in __post_init__) ─────────────────────────────
    checkpoint_dir: str = ""
    results_dir: str = ""

    def __post_init__(self) -> None:
        root = Path(self.output_root)
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(root / "checkpoints")
        if not self.results_dir:
            self.results_dir = str(root / "results")

    @property
    def partition_dir(self) -> Path:
        """Writable directory for hospital partition JSON files."""
        return Path(self.output_root) / "partitions"


# ── Class metadata ────────────────────────────────────────────────────────────

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

DX_TO_CLASS = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AK",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}

CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
}

CLINICAL_NOTES = {
    "MEL": (
        "Melanoma is the most dangerous form of skin cancer. "
        "Early detection is critical for successful treatment."
    ),
    "NV": (
        "Melanocytic Nevi (moles) are common benign skin lesions. "
        "Most are harmless but should be monitored for changes."
    ),
    "BCC": (
        "Basal Cell Carcinoma is the most common form of skin cancer. "
        "It rarely spreads but requires treatment."
    ),
    "AK": (
        "Actinic Keratosis is a pre-cancerous rough skin patch caused by sun damage. "
        "Treatment can prevent progression to SCC."
    ),
    "BKL": (
        "Benign Keratosis includes seborrheic keratoses and solar lentigines. "
        "Generally harmless growths that may be removed for cosmetic reasons."
    ),
    "DF": (
        "Dermatofibroma is a benign fibrous nodule common on the legs. "
        "Usually harmless and may not require treatment."
    ),
    "VASC": (
        "Vascular Lesions include angiomas and pyogenic granulomas. "
        "Most are benign; treatment depends on symptoms and location."
    ),
}

HOSPITAL_IDS = ["A", "B", "C"]

# Kaggle dataset slug — used by scripts to build the input path automatically
KAGGLE_DATA_ROOT = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
KAGGLE_OUTPUT_ROOT = "/kaggle/working"
