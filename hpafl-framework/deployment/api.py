"""FastAPI inference server for the HPAFL global model.

Exposes endpoints for health check, metrics retrieval, image prediction,
and adaptive weight queries.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from config import CLASS_FULL_NAMES, CLASS_NAMES, CLINICAL_NOTES, HAPFLConfig
from models.efficientnet_model import get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App and CORS setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HPAFL Medical Image Classifier",
    version="1.0.0",
    description=(
        "Privacy-aware federated skin lesion classifier trained on HAM10000. "
        "For research purposes only — not a clinical diagnosis tool."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: Optional[nn.Module] = None
_config: HAPFLConfig = HAPFLConfig()
_model_version: str = "unloaded"
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    predicted_class: str
    predicted_class_code: str
    confidence: float
    top3_predictions: List[Dict]
    clinical_note: str
    model_version: str
    disclaimer: str = "For research purposes only. Not a clinical diagnosis."


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str
    model_version: str
    last_updated: str
    num_rounds_trained: int
    device: str


class MetricsResponse(BaseModel):
    """Response schema for the /metrics endpoint."""

    accuracy: float
    f1_macro: float
    roc_auc_ovr: float
    epsilon: float


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def load_model() -> None:
    """Load the global model checkpoint on server startup."""
    global _model, _model_version

    checkpoint_path = Path(_config.checkpoint_dir) / "global_model_final.pt"

    _model = get_model(_config)

    if checkpoint_path.exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location=_device)
            _model.load_state_dict(state_dict, strict=False)
            _model_version = "global_model_final"
            logger.info("Loaded model from %s", checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to load checkpoint (%s). Using random weights.", exc)
            _model_version = "random_weights"
    else:
        logger.warning(
            "No checkpoint found at %s. Using random weights for inference.", checkpoint_path
        )
        _model_version = "random_weights"

    _model.to(_device)
    _model.eval()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with server status, model version, and device info.
    """
    import datetime

    results_path = Path(_config.results_dir) / "hpafl.json"
    num_rounds = 0
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        num_rounds = data.get("num_rounds_trained", 0)

    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model_version=_model_version,
        last_updated=datetime.datetime.now().isoformat(),
        num_rounds_trained=num_rounds,
        device=str(_device),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return latest evaluation metrics from results/hpafl.json.

    Returns:
        MetricsResponse with accuracy, F1, ROC-AUC, and privacy epsilon.

    Raises:
        HTTPException: If results file is not found.
    """
    results_path = Path(_config.results_dir) / "hpafl.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Results not found. Run the HPAFL training pipeline first.",
        )
    with open(results_path) as f:
        data = json.load(f)
    return MetricsResponse(
        accuracy=data.get("accuracy", 0.0),
        f1_macro=data.get("f1_macro", 0.0),
        roc_auc_ovr=data.get("roc_auc_ovr", 0.0),
        epsilon=data.get("epsilon", 0.0),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Accept a dermoscopic image and return a skin lesion classification.

    Args:
        file: Uploaded image file (JPEG/PNG).

    Returns:
        PredictionResponse with predicted class, confidence, and top-3 predictions.

    Raises:
        HTTPException: If model is not loaded or image cannot be processed.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Read and preprocess image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")

    try:
        tensor = _VAL_TRANSFORM(image).unsqueeze(0).to(_device)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {exc}")

    # Inference
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class": CLASS_FULL_NAMES[CLASS_NAMES[i]],
            "class_code": CLASS_NAMES[i],
            "confidence": float(round(probs[i], 4)),
        }
        for i in top3_idx
    ]

    best_idx = int(np.argmax(probs))
    best_code = CLASS_NAMES[best_idx]

    return PredictionResponse(
        predicted_class=CLASS_FULL_NAMES[best_code],
        predicted_class_code=best_code,
        confidence=float(round(probs[best_idx], 4)),
        top3_predictions=top3,
        clinical_note=CLINICAL_NOTES.get(best_code, ""),
        model_version=_model_version,
    )


@app.get("/weights")
async def adaptive_weights() -> Dict:
    """Return latest adaptive aggregation weights per hospital per round.

    Used by the Streamlit dashboard to render weight evolution charts.

    Returns:
        Dict with rounds list and per-hospital weight history.

    Raises:
        HTTPException: If the adaptive weights CSV is not found.
    """
    import pandas as pd

    csv_path = Path(_config.results_dir) / "adaptive_weights.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Adaptive weights CSV not found. Run training first.",
        )
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deployment.api:app",
        host=_config.api_host,
        port=_config.api_port,
        reload=False,
        log_level="info",
    )
