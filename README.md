# HPAFL - Hybrid Privacy-Aware Federated Learning Framework

A production-grade federated learning system for skin lesion classification on the HAM10000 dataset. HPAFL simultaneously provides differential privacy (DP-SGD), secure model aggregation, adaptive weighted aggregation, and feature-shift robustness via FedBN - demonstrating measurable improvements over the standard FedAvg baseline while preserving patient data privacy across multiple hospital nodes.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    HPAFL Architecture                     │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │ Hospital A  │  │ Hospital B  │  │ Hospital C  │      │
│   │             │  │             │  │             │      │
│   │ EfficientB0 │  │ EfficientB0 │  │ EfficientB0 │      │
│   │ DP-SGD      │  │ DP-SGD      │  │ DP-SGD      │      │
│   │ FocalLoss   │  │ FocalLoss   │  │ FocalLoss   │      │
│   │ Local BN ✓  │  │ Local BN ✓  │  │ Local BN ✓  │      │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│          │   Masked params │ (SecureAgg)    │             │
│          └────────────────┬────────────────┘             │
│                           ▼                               │
│               ┌───────────────────────┐                   │
│               │   FL Server           │                   │
│               │   Adaptive Weighted   │                   │
│               │   Aggregation         │                   │
│               │   (replaces FedAvg)   │                   │
│               └───────────┬───────────┘                   │
│                           ▼                               │
│               ┌───────────────────────┐                   │
│               │  Global Model         │                   │
│               │  FastAPI Inference    │                   │
│               │  Streamlit Dashboard  │                   │
│               └───────────────────────┘                   │
└───────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 2: Prepare data partitions**

```bash
# Point to your HAM10000 archive directory
python data/prepare_data.py
```

**Step 3: Run HPAFL training**

```bash
# Full HPAFL pipeline (DP + SecureAgg + AdaptiveAgg + FedBN)
python scripts/run_hpafl.py

# Or run all baselines + report
python scripts/benchmark.py
```

**Step 4: Launch the API and dashboard**

```bash
# API server
uvicorn deployment.api:app --host 0.0.0.0 --port 8000

# Dashboard (separate terminal)
streamlit run deployment/dashboard.py
```

---

## Hyperparameter Table

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 7 | Number of skin lesion classes |
| `image_size` | 224 | Input image size (pixels, square) |
| `num_hospitals` | 3 | Number of federated hospital clients |
| `dirichlet_alpha` | 0.5 | Non-IID concentration parameter (lower = more skew) |
| `model_name` | efficientnet_b0 | EfficientNet variant |
| `pretrained` | True | Load ImageNet pretrained weights |
| `dropout_rate` | 0.3 | Dropout in classification head |
| `hidden_dim` | 512 | Hidden layer size in classifier head |
| `local_epochs` | 5 | Local training epochs per FL round |
| `batch_size` | 32 | Mini-batch size |
| `learning_rate` | 1e-4 | Initial learning rate (decays 0.95/round) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `num_rounds` | 20 | Total FL communication rounds |
| `noise_multiplier` | 1.1 | Gaussian noise for DP-SGD |
| `max_grad_norm` | 1.0 | Per-sample gradient clipping bound |
| `target_epsilon` | 8.0 | Privacy budget upper bound (ε) |
| `target_delta` | 1e-5 | Privacy failure probability (δ) |
| `alpha_accuracy` | 0.40 | Accuracy weight in adaptive score |
| `alpha_reliability` | 0.30 | Reliability weight in adaptive score |
| `alpha_data_quality` | 0.20 | Data quality weight in adaptive score |
| `alpha_historical` | 0.10 | Historical EMA weight in adaptive score |
| `ema_decay` | 0.30 | EMA decay for historical accuracy tracking |

---

## Expected Results

| System | Accuracy | F1-Macro | ROC-AUC | Privacy (ε) | Security |
|--------|----------|----------|---------|-------------|----------|
| Centralised (ceiling) | ~0.84 | ~0.72 | ~0.91 | ∞ | None |
| FedAvg (McMahan 2017) | ~0.76 | ~0.64 | ~0.85 | ∞ | None |
| FedProx + DP | ~0.74 | ~0.62 | ~0.83 | ≤ 8.0 | None |
| **HPAFL (ours)** | **~0.80** | **~0.69** | **~0.88** | **≤ 8.0** | **SecureAgg** |

*Results are approximate and depend on hardware, random seeds, and data version.*

---

## Project Structure

```
hpafl-framework/
├── config.py               # All hyperparameters (single source of truth)
├── requirements.txt
├── data/
│   ├── prepare_data.py     # Dirichlet non-IID hospital partitioning
│   └── dataset.py          # HAM10000Dataset with augmentation
├── models/
│   ├── efficientnet_model.py  # EfficientNet-B0 + custom head
│   └── focal_loss.py          # Focal Loss for class imbalance
├── privacy/
│   └── dp_trainer.py          # Opacus DP-SGD wrapper
├── security/
│   └── secure_agg.py          # Bonawitz et al. SecureAgg simulation
├── clients/
│   ├── local_trainer.py       # Local DP training loop
│   └── client.py              # Flower NumPyClient (FedBN)
├── server/
│   ├── adaptive_agg.py        # Multi-factor adaptive aggregation strategy
│   └── server.py              # Flower server setup
├── evaluation/
│   └── evaluate.py            # Metrics, plots, baseline comparison
├── deployment/
│   ├── api.py                 # FastAPI inference server
│   └── dashboard.py           # Streamlit monitoring dashboard
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
└── docker-compose.yml
```

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services started:
#   flower-server     → port 8080 (Flower FL server)
#   hospital-a/b/c   → FL clients (hospitals)
#   api-server        → port 8000 (FastAPI inference)
#   streamlit-dashboard → port 8501 (monitoring UI)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Citations

```bibtex
@inproceedings{mcmahan2017fedavg,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Agüera},
  booktitle={AISTATS},
  year={2017}
}

@inproceedings{li2021fedbn,
  title={FedBN: Federated Learning on Non-IID Features via Local Batch Normalization},
  author={Li, Xiaoxiao and Jiang, Meirui and Zhang, Xiaofei and Kamp, Michael and Dou, Qi},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{abadi2016dpsgd,
  title={Deep learning with differential privacy},
  author={Abadi, Martín and others},
  booktitle={ACM CCS},
  year={2016}
}

@inproceedings{bonawitz2017secagg,
  title={Practical secure aggregation for privacy-preserving machine learning},
  author={Bonawitz, Keith and others},
  booktitle={ACM CCS},
  year={2017}
}
```