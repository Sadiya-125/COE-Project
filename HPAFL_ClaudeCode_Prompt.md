# Claude Code System Prompt
## Hybrid Privacy-Aware Federated Learning Framework (HPAFL)
### Copy this entire prompt into Claude Code to build the complete project

---

You are a Senior ML Engineer implementing a production-grade **Hybrid Privacy-Aware Federated Learning Framework (HPAFL)** for medical image classification on HAM10000. Build the COMPLETE project as described below. Every file must be fully functional, documented with Google-style docstrings, and immediately executable. Do not write stubs or placeholders — write the complete working code for each file before moving to the next.

---

## PROJECT CONTEXT

**Goal:** Federated skin lesion classification (7 classes, HAM10000 dataset) that is demonstrably better than standard FedAvg. The system must simultaneously provide:
- Privacy via Opacus DP-SGD (ε < 8, δ = 1e-5)
- Security via Secure Aggregation (pairwise masking simulation)
- Better aggregation via Adaptive Weighted Strategy (replaces FedAvg)
- Feature-shift robustness via FedBN (local BN layers not aggregated)
- Production deployment via FastAPI + Streamlit + Docker

**Academic baseline papers to exceed:**
- FedAvg (McMahan et al., AISTATS 2017) — the standard baseline
- FedBN (Li et al., ICLR 2021) — our BN-exclusion strategy inspiration
- DP-SGD (Abadi et al., CCS 2016) — our privacy mechanism
- Secure Aggregation (Bonawitz et al., CCS 2017) — our security layer

---

## STEP 1 — PROJECT SCAFFOLD

Create this exact directory structure first:

```
hpafl-framework/
├── config.py
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── prepare_data.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── efficientnet_model.py
│   └── focal_loss.py
├── privacy/
│   ├── __init__.py
│   └── dp_trainer.py
├── security/
│   ├── __init__.py
│   └── secure_agg.py
├── clients/
│   ├── __init__.py
│   ├── client.py
│   └── local_trainer.py
├── server/
│   ├── __init__.py
│   ├── server.py
│   └── adaptive_agg.py
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
├── deployment/
│   ├── api.py
│   └── dashboard.py
├── scripts/
│   ├── run_centralized.py
│   ├── run_fedavg.py
│   ├── run_hpafl.py
│   └── benchmark.py
├── tests/
│   ├── test_model.py
│   ├── test_privacy.py
│   ├── test_aggregation.py
│   └── test_client.py
├── docker/
│   ├── Dockerfile.client
│   ├── Dockerfile.server
│   └── Dockerfile.api
├── docker-compose.yml
└── README.md
```

---

## STEP 2 — requirements.txt

```
flwr==1.7.0
torch>=2.1.0
torchvision>=0.16.0
opacus>=1.4.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
streamlit>=1.33.0
plotly>=5.20.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
tqdm>=4.66.0
python-dotenv>=1.0.0
requests>=2.31.0
efficientnet-pytorch>=0.7.1
```

---

## STEP 3 — config.py

Create a `HAPFLConfig` dataclass with ALL hyperparameters. No hardcoded values anywhere else in the codebase — always import from config. Include:

```python
@dataclass
class HAPFLConfig:
    # Data
    data_root: str = "./data/ham10000"
    num_classes: int = 7
    image_size: int = 224
    num_hospitals: int = 3
    hospital_splits: tuple = (0.35, 0.35, 0.30)
    dirichlet_alpha: float = 0.5
    
    # Model
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout_rate: float = 0.3
    hidden_dim: int = 512
    
    # Training
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    focal_gamma: float = 2.0
    
    # Federated Learning
    num_rounds: int = 20
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    server_address: str = "0.0.0.0:8080"
    
    # Differential Privacy
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    
    # Adaptive Aggregation weights (must sum to 1.0)
    alpha_accuracy: float = 0.40
    alpha_reliability: float = 0.30
    alpha_data_quality: float = 0.20
    alpha_historical: float = 0.10
    ema_decay: float = 0.30
    
    # Deployment
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]
CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion"
}
```

---

## STEP 4 — data/prepare_data.py

Implement `DatasetPartitioner` that:

1. **Loads HAM10000** — reads `HAM10000_metadata.csv`, maps `dx` column to integer labels 0–6 using `CLASS_NAMES` order
2. **Dirichlet non-IID split** — uses `np.random.dirichlet(alpha * np.ones(num_hospitals))` per class to assign sample indices, creating realistic label distribution skew
3. **Intentional skew** — after Dirichlet split, apply secondary bias: Hospital A gets extra melanoma/nevi samples, Hospital B gets extra BCC/AK, Hospital C gets extra DF/VASC (swap 15% of samples to reinforce skew)
4. **Saves** partition indices as `data/partitions/hospital_{A,B,C}_indices.json`
5. **Prints** partition statistics: class distribution per hospital, total counts, imbalance ratio

Also implement `verify_partitions()` that:
- Confirms no overlap between hospital splits
- Prints a distribution table
- Saves a `data/partitions/partition_stats.json` summary

---

## STEP 5 — data/dataset.py

Implement `HAM10000Dataset(torch.utils.data.Dataset)`:

```python
class HAM10000Dataset(Dataset):
    """HAM10000 dermoscopic image dataset with configurable transforms."""
    
    def __init__(self, data_root, indices, split="train", config=None):
        # split: "train" or "val"
        # For train: full augmentation pipeline
        # For val: only resize + normalize
        ...
    
    @property
    def class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for WeightedRandomSampler."""
        ...
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Return WeightedRandomSampler to handle class imbalance."""
        ...
```

**Training transforms:** `Resize(224)` → `RandomHorizontalFlip(0.5)` → `RandomVerticalFlip(0.3)` → `RandomRotation(15)` → `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)` → `RandomAffine(degrees=0, translate=(0.1, 0.1))` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**Val transforms:** `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(...)`

Implement `create_dataloaders(hospital_id, config)` factory function returning `(train_loader, val_loader)`.

---

## STEP 6 — models/focal_loss.py

Implement `FocalLoss(nn.Module)`:

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification.
    
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    Args:
        gamma: Focusing parameter. 0 = standard cross entropy. Default: 2.0
        alpha: Class weights tensor of shape (num_classes,). Default: None (uniform)
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """
```

---

## STEP 7 — models/efficientnet_model.py

Implement:

```python
def get_model(config: HAPFLConfig) -> nn.Module:
    """Load EfficientNet-B0 with custom classification head.
    
    Architecture:
        EfficientNet-B0 backbone (pretrained on ImageNet)
        → features (1280-dim)
        → Linear(1280, 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
        → Linear(512, 7)
    
    BN layers are identifiable via get_bn_layer_names() for FedBN exclusion.
    """

def get_bn_layer_names(model: nn.Module) -> list[str]:
    """Return list of all BatchNorm layer parameter names for FedBN exclusion."""

def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze/unfreeze EfficientNet backbone, keep classifier trainable."""

def count_parameters(model: nn.Module) -> dict:
    """Return dict with total, trainable, frozen parameter counts."""
```

---

## STEP 8 — privacy/dp_trainer.py

Implement `DPTrainer`:

```python
class DPTrainer:
    """Wraps Opacus PrivacyEngine for differentially private local training.
    
    Attaches to a model+optimizer+dataloader triplet and tracks cumulative
    (epsilon, delta) privacy budget across all training steps.
    """
    
    def __init__(self, config: HAPFLConfig):
        ...
    
    def attach(self, model, optimizer, dataloader):
        """Attach Opacus PrivacyEngine. Must be called before training."""
        # Returns (dp_model, dp_optimizer, dp_dataloader)
        ...
    
    def get_epsilon(self) -> float:
        """Return current cumulative epsilon spend."""
        ...
    
    def is_budget_exhausted(self) -> bool:
        """Return True if epsilon >= target_epsilon."""
        ...
    
    def get_privacy_report(self) -> dict:
        """Return {'epsilon': float, 'delta': float, 'noise_multiplier': float, 
                   'max_grad_norm': float, 'budget_remaining': float}"""
        ...
```

Also implement `PrivacyBudgetLogger` that writes per-round epsilon to `results/privacy_budget.csv`.

---

## STEP 9 — security/secure_agg.py

Implement a simulation of the Bonawitz et al. secure aggregation protocol:

```python
class SecureAggregator:
    """Simulates the Bonawitz et al. (CCS 2017) Secure Aggregation protocol.
    
    In a real deployment this would use actual cryptographic primitives
    (Diffie-Hellman key agreement, Shamir secret sharing). This simulation
    demonstrates the protocol structure and correctness:
    - Each pair of clients agrees on a shared random mask
    - Each client adds all pairwise masks (cancelling in sum) + a self-mask
    - Server only ever sees the aggregate, never individual updates
    
    Args:
        num_clients: Number of participating clients
        threshold: Minimum clients needed for reconstruction (t-out-of-n)
    """
    
    def mask_update(self, client_id: int, update: list[np.ndarray], 
                    round_num: int) -> list[np.ndarray]:
        """Apply pairwise masks + self-mask to client update."""
        ...
    
    def aggregate(self, masked_updates: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
        """Reconstruct the true aggregate from masked updates.
        Verify that individual updates cannot be recovered from masked_updates alone.
        """
        ...
    
    def verify_security(self, masked_updates, true_aggregate) -> bool:
        """Assert that no individual update is recoverable from masked_updates."""
        ...
```

---

## STEP 10 — clients/local_trainer.py

Implement `LocalTrainer`:

```python
class LocalTrainer:
    """Manages local model training with DP-SGD and FocalLoss.
    
    Trains for config.local_epochs epochs, applies DP-SGD via DPTrainer,
    and returns updated parameters with training metrics.
    """
    
    def train(self, model, train_loader, val_loader) -> dict:
        """Run local training. Returns:
        {
            'loss': float,          # final training loss
            'accuracy': float,      # validation accuracy
            'f1_macro': float,      # macro F1 on validation
            'epsilon': float,       # cumulative privacy budget spent
            'num_samples': int,
            'epochs_trained': int,
        }
        """
        ...
    
    def _train_epoch(self, model, loader, optimizer, criterion) -> float:
        """Single training epoch. Returns average loss."""
        ...
    
    def _validate(self, model, loader) -> dict:
        """Validation pass. Returns accuracy, f1, per-class metrics."""
        ...
```

---

## STEP 11 — clients/client.py

Implement `HAM10000FlowerClient(fl.client.NumPyClient)`:

```python
class HAM10000FlowerClient(fl.client.NumPyClient):
    """Flower client for one hospital node.
    
    Key FedBN behaviour: get_parameters() EXCLUDES BN layer params.
    set_parameters() updates ONLY non-BN layers.
    BN layers are trained locally and never shared.
    """
    
    def __init__(self, hospital_id: str, config: HAPFLConfig):
        ...
    
    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return non-BN model parameters as numpy arrays."""
        ...
    
    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set non-BN model parameters from numpy arrays. BN layers unchanged."""
        ...
    
    def fit(self, parameters, config) -> tuple[list, int, dict]:
        """Local training round.
        Returns: (updated_params, num_examples, 
                  {'loss': float, 'accuracy': float, 'epsilon': float, 
                   'hospital_id': str, 'round': int})
        """
        ...
    
    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Local evaluation.
        Returns: (loss, num_examples, 
                  {'accuracy': float, 'f1_macro': float, 'roc_auc': float,
                   'per_class_f1': dict, 'hospital_id': str})
        """
        ...

def start_client(hospital_id: str, server_address: str, config: HAPFLConfig):
    """Start a Flower client for the given hospital."""
    fl.client.start_numpy_client(
        server_address=server_address,
        client=HAM10000FlowerClient(hospital_id, config),
    )
```

---

## STEP 12 — server/adaptive_agg.py

Implement `AdaptiveAggregationStrategy(fl.server.strategy.FedAvg)`:

```python
class AdaptiveAggregationStrategy(fl.server.strategy.FedAvg):
    """Replaces FedAvg sample-proportion weighting with multi-factor adaptive scoring.
    
    Composite score per client k:
        Sₖ = Accₖ^α_acc × Relₖ^α_rel × DQₖ^α_dq × Hₖ^α_hist
    
    Where:
        Accₖ  = validation accuracy this round
        Relₖ  = rounds participated / total rounds (reliability)
        DQₖ   = num_samples / max_samples × class_diversity_score
        Hₖ    = EMA of historical accuracy (decay = 0.3)
    
    Normalised weight: wₖ = Sₖ / Σⱼ Sⱼ
    """
    
    def __init__(self, config: HAPFLConfig, secure_aggregator: SecureAggregator, **kwargs):
        ...
    
    def aggregate_fit(self, server_round, results, failures):
        """Override FedAvg aggregate_fit with adaptive weights + secure aggregation."""
        # 1. Extract masked updates via SecureAggregator
        # 2. Compute adaptive weights per client
        # 3. Apply weighted average
        # 4. Log weights to CSV
        # 5. Return aggregated parameters
        ...
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Weighted evaluation metrics aggregation."""
        ...
    
    def compute_adaptive_weight(self, client_id, accuracy, num_examples, round_num) -> float:
        """Compute composite score and return normalised weight for one client."""
        ...
    
    def _update_history(self, client_id: str, accuracy: float) -> None:
        """Update EMA history for a client."""
        ...
    
    def save_round_weights(self, server_round: int, weights: dict) -> None:
        """Append per-client weights for this round to results/adaptive_weights.csv."""
        ...
```

---

## STEP 13 — server/server.py

```python
def create_strategy(config: HAPFLConfig) -> AdaptiveAggregationStrategy:
    """Build the aggregation strategy with all components wired together."""
    ...

def run_server(config: HAPFLConfig) -> fl.server.History:
    """Start the Flower federated learning server.
    
    Configures:
    - AdaptiveAggregationStrategy
    - ServerConfig(num_rounds=config.num_rounds)  
    - on_fit_config_fn: sends round number, learning rate, DP config to clients
    - on_evaluate_config_fn: sends eval config
    - Saves final global model to config.checkpoint_dir/global_model_final.pt
    """
    ...

def on_fit_config(server_round: int) -> dict:
    """Return per-round fit config dict sent to each client."""
    return {
        "round": server_round,
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate * (0.95 ** server_round),  # LR decay
        "noise_multiplier": config.noise_multiplier,
        "max_grad_norm": config.max_grad_norm,
    }
```

---

## STEP 14 — evaluation/evaluate.py

Implement:

```python
def evaluate_model(model, dataloader, class_names, device) -> dict:
    """Full evaluation returning accuracy, precision, recall, f1_macro, 
    f1_per_class, roc_auc_ovr, confusion_matrix."""
    ...

def compare_baselines(results_dir: str) -> pd.DataFrame:
    """Load results JSON files for all baselines, return comparison DataFrame."""
    # Looks for: results/centralized.json, results/fedavg.json, 
    #            results/fedprox_dp.json, results/hpafl.json
    ...

def plot_learning_curves(results_dir: str) -> None:
    """Plotly line chart: accuracy per hospital per FL round. 
    Saves to results/plots/learning_curves.html"""
    ...

def plot_privacy_budget(results_dir: str) -> None:
    """Plotly line chart: cumulative epsilon per hospital per round.
    Saves to results/plots/privacy_budget.html"""
    ...

def plot_adaptive_weights(results_dir: str) -> None:
    """Plotly stacked bar: per-client adaptive weights per round.
    Saves to results/plots/adaptive_weights.html"""
    ...

def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str) -> None:
    """Plotly heatmap confusion matrix with class names.
    Saves to results/plots/confusion_matrix.html"""
    ...

def generate_full_report(results_dir: str) -> None:
    """Run all evaluation functions and print formatted comparison table to stdout."""
    ...
```

---

## STEP 15 — deployment/api.py

```python
"""FastAPI inference server for the HPAFL global model."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HPAFL Medical Image Classifier", version="1.0.0")

# Endpoints:
# GET  /health      → {status, model_version, last_updated, num_rounds_trained}
# GET  /metrics     → latest evaluation metrics from results/hpafl.json
# POST /predict     → accepts image file, returns prediction + confidence
# GET  /weights     → latest adaptive weights per hospital per round (for dashboard)

class PredictionResponse(BaseModel):
    predicted_class: str           # e.g. "Melanoma"
    predicted_class_code: str      # e.g. "MEL"
    confidence: float              # 0.0 – 1.0
    top3_predictions: list[dict]   # [{"class": str, "confidence": float}] × 3
    clinical_note: str             # Brief explanation of the predicted class
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Accept a dermoscopic image and return classification result."""
    ...

# Include startup event that loads model from checkpoint_dir/global_model_final.pt
```

---

## STEP 16 — deployment/dashboard.py

Build a Streamlit multi-page dashboard:

```python
"""Streamlit monitoring and inference dashboard for HPAFL."""

# PAGE LAYOUT: Use st.sidebar.radio for page navigation

# PAGE 1: "Training Monitor"
# - st.header("Federated Training Progress")
# - Row 1: 3 metric cards showing latest accuracy per hospital (st.metric)
# - Row 2: Plotly line chart — accuracy per hospital per round (read from results/adaptive_weights.csv)
# - Row 3: Plotly stacked bar — adaptive weight evolution per round
# - Row 4: Plotly line — privacy budget (epsilon) per hospital per round
# - st.button("Refresh") to reload data

# PAGE 2: "Model Evaluation"  
# - Baseline comparison table (st.dataframe with highlighting)
# - Confusion matrix heatmap (plotly)
# - Per-class F1 bar chart
# - ROC curve per class (one-vs-rest)

# PAGE 3: "Live Inference"
# - st.file_uploader("Upload a dermoscopic skin image")
# - On upload: call FastAPI /predict endpoint, display:
#   - Image preview (st.image)
#   - Top prediction with confidence bar (st.progress)
#   - Top-3 predictions table
#   - Clinical note / class explanation
#   - Disclaimer: "For research purposes only. Not a clinical diagnosis."
```

---

## STEP 17 — scripts/

**run_centralized.py:** Train EfficientNet-B0 on the full HAM10000 dataset (no federation). Save results to `results/centralized.json`. This is the performance ceiling.

**run_fedavg.py:** Standard Flower FedAvg, no DP, no secure agg, no adaptive weights. Save results to `results/fedavg.json`.

**run_hpafl.py:** Full HPAFL with all components. Start server in a thread, start 3 clients in separate processes, run 20 rounds. Save results to `results/hpafl.json`.

**benchmark.py:** Runs all three scripts sequentially, then calls `generate_full_report()` and prints a formatted comparison table.

---

## STEP 18 — Docker

**docker/Dockerfile.client:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/run_client.py"]
```

**docker/Dockerfile.server:** Similar, CMD starts the Flower server.

**docker-compose.yml:** 6 services — flower-server (port 8080), hospital-a, hospital-b, hospital-c (all depend on flower-server), api-server (port 8000, depends on flower-server), streamlit-dashboard (port 8501, depends on api-server). All on the same bridge network `fl-network`. Shared volume `./checkpoints:/checkpoints`.

---

## STEP 19 — tests/

Write pytest tests for:

**test_model.py:**
- `test_model_output_shape()` — forward pass returns (batch, 7)
- `test_bn_layer_detection()` — `get_bn_layer_names()` returns non-empty list
- `test_focal_loss_values()` — FL > 0, FL < CE for easy examples

**test_privacy.py:**
- `test_dp_trainer_attaches()` — DPTrainer.attach() succeeds
- `test_epsilon_increases()` — epsilon increases monotonically with steps
- `test_budget_exhausted()` — `is_budget_exhausted()` returns True when ε≥target

**test_aggregation.py:**
- `test_adaptive_weights_sum_to_one()` — weights sum to 1.0
- `test_high_accuracy_gets_higher_weight()` — client with higher accuracy gets higher weight
- `test_secure_agg_correctness()` — SecureAggregator.aggregate() equals true sum

**test_client.py:**
- `test_get_parameters_excludes_bn()` — BN params absent from returned list
- `test_set_parameters_preserves_bn()` — set_parameters doesn't change BN layer values

---

## STEP 20 — README.md

Write a complete README with:
1. One-paragraph project description
2. ASCII architecture diagram (text-based)
3. Quick-start commands (3 steps: install, prepare data, run)
4. Hyperparameter table (all config values with descriptions)
5. Expected results table (4 systems × 6 metrics)
6. Project structure tree
7. Citation section (McMahan 2017, Li 2021 FedBN, Abadi 2016, Bonawitz 2017)
8. Troubleshooting section (common errors + fixes)

---

## IMPLEMENTATION RULES

1. **Every file** must have a module-level docstring explaining its role in the system
2. **Every function** must have Google-style docstrings with Args, Returns, Raises
3. **All types** annotated with Python type hints
4. **All hyperparameters** imported from `config.py` — no magic numbers
5. **All logging** via `logging.getLogger(__name__)` — no bare `print()` calls
6. **No hardcoded paths** — always use `config.data_root`, `config.checkpoint_dir`, etc.
7. **Error handling** with meaningful messages; never silent `except: pass`
8. Build in this exact order: config → data → models → privacy → security → clients → server → evaluation → deployment → scripts → tests → README

## VERIFICATION

After all files are created, run:
```bash
# Verify imports
python -c "import flwr, opacus, torch, fastapi, streamlit; print('All imports OK')"

# Run tests
pytest tests/ -v

# Verify data pipeline (with dummy data)
python -c "from data.dataset import HAM10000Dataset; print('Dataset OK')"

# Verify model
python -c "from models.efficientnet_model import get_model; from config import HAPFLConfig; m=get_model(HAPFLConfig()); print('Model OK, params:', sum(p.numel() for p in m.parameters()))"
```

If any verification fails, fix it before proceeding to the next step.
