"""Streamlit monitoring and inference dashboard for HPAFL.

Three pages:
    1. Training Monitor — per-hospital accuracy, weight evolution, privacy budget.
    2. Model Evaluation — baseline comparison, confusion matrix, F1 per class.
    3. Live Inference — upload an image and get a classification with clinical note.
"""

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import CLASS_NAMES, HAPFLConfig, KAGGLE_OUTPUT_ROOT, is_kaggle

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HPAFL Dashboard",
    page_icon="\U0001f3e5",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = HAPFLConfig()
if is_kaggle():
    cfg.output_root = KAGGLE_OUTPUT_ROOT
else:
    cfg.output_root = str(Path(__file__).resolve().parents[1])

API_BASE = f"http://localhost:{cfg.api_port}"
RESULTS_DIR = Path(cfg.results_dir)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["Training Monitor", "Model Evaluation", "Live Inference"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**HPAFL** — Hybrid Privacy-Aware Federated Learning\n\n"
    "For research purposes only."
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_weights_csv() -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / "adaptive_weights.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_privacy_csv() -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / "privacy_budget.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_results(fname: str) -> Optional[Dict]:
    p = RESULTS_DIR / fname
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _chart(fig):
    """Render a Plotly figure full-width (Streamlit 1.33+ API)."""
    try:
        st.plotly_chart(fig, width="stretch")   # Streamlit >= 1.33 deprecates use_container_width
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)  # fallback for older versions


# ---------------------------------------------------------------------------
# Page 1: Training Monitor
# ---------------------------------------------------------------------------

if page == "Training Monitor":
    st.header("Federated Training Progress")

    if st.button("Refresh Data"):
        st.rerun()  # was st.experimental_rerun() in older Streamlit

    df_weights = _load_weights_csv()

    st.subheader("Latest Accuracy per Hospital")
    if df_weights is not None and not df_weights.empty:
        latest_round = df_weights["round"].max()
        latest = df_weights[df_weights["round"] == latest_round]
        cols = st.columns(len(latest))
        for col, (_, row) in zip(cols, latest.iterrows()):
            col.metric(
                label=f"Hospital {row['hospital_id']}",
                value=f"{row['accuracy']:.1%}",
                delta=f"weight: {row['weight']:.3f}",
            )
    else:
        st.info("No training data found. Run the HPAFL training pipeline first.")

    if df_weights is not None and not df_weights.empty:
        st.subheader("Accuracy per Hospital per FL Round")
        fig_acc = px.line(
            df_weights, x="round", y="accuracy", color="hospital_id", markers=True,
            title="Validation Accuracy per Hospital",
            labels={"round": "FL Round", "accuracy": "Accuracy", "hospital_id": "Hospital"},
        )
        fig_acc.update_layout(template="plotly_white")
        _chart(fig_acc)

        st.subheader("Adaptive Aggregation Weight Evolution")
        fig_w = px.bar(
            df_weights, x="round", y="weight", color="hospital_id", barmode="stack",
            title="Normalised Adaptive Weights per Round",
            labels={"round": "FL Round", "weight": "Weight", "hospital_id": "Hospital"},
        )
        fig_w.update_layout(template="plotly_white")
        _chart(fig_w)

    df_privacy = _load_privacy_csv()
    if df_privacy is not None and not df_privacy.empty:
        st.subheader("Privacy Budget Consumption (Cumulative epsilon)")
        fig_priv = px.line(
            df_privacy, x="round", y="epsilon", color="hospital_id", markers=True,
            title="Cumulative Privacy Budget per Hospital",
            labels={"round": "FL Round", "epsilon": "Cumulative epsilon", "hospital_id": "Hospital"},
        )
        fig_priv.add_hline(
            y=cfg.target_epsilon, line_dash="dash", line_color="red",
            annotation_text=f"Budget limit = {cfg.target_epsilon}",
        )
        fig_priv.update_layout(template="plotly_white")
        _chart(fig_priv)


# ---------------------------------------------------------------------------
# Page 2: Model Evaluation
# ---------------------------------------------------------------------------

elif page == "Model Evaluation":
    st.header("Model Evaluation & Baseline Comparison")

    systems_map = {
        "Centralized": "centralized.json",
        "FedAvg": "fedavg.json",
        "FedProx+DP": "fedprox_dp.json",
        "HPAFL (ours)": "hpafl.json",
    }

    rows = []
    for sname, fname in systems_map.items():
        data = _load_results(fname)
        if data:
            rows.append({
                "System": sname,
                "Accuracy": f"{data.get('accuracy', 0):.4f}",
                "F1-Macro": f"{data.get('f1_macro', 0):.4f}",
                "ROC-AUC": f"{data.get('roc_auc_ovr', 0):.4f}",
                "Precision": f"{data.get('precision_macro', 0):.4f}",
                "Recall": f"{data.get('recall_macro', 0):.4f}",
                "epsilon (privacy)": f"{data.get('epsilon', float('inf')):.2f}",
            })

    if rows:
        st.subheader("Baseline Comparison Table")
        df_cmp = pd.DataFrame(rows)
        st.dataframe(
            df_cmp.style.highlight_max(
                subset=["Accuracy", "F1-Macro", "ROC-AUC"], color="lightgreen",
            ),
            use_container_width=True,  # noqa: deprecated in 1.58 but st.dataframe keeps it
        )
    else:
        st.info("No results files found. Run training scripts first.")

    hpafl_data = _load_results("hpafl.json")
    if hpafl_data and "confusion_matrix" in hpafl_data:
        st.subheader("HPAFL Confusion Matrix")
        cm_arr = np.array(hpafl_data["confusion_matrix"], dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_arr / row_sums, 0.0)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_norm, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues",
            text=[[str(int(cm_arr[r][c])) for c in range(len(CLASS_NAMES))]
                  for r in range(len(CLASS_NAMES))],
            texttemplate="%{text}",
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="True",
            title="Confusion Matrix (row-normalised)", template="plotly_white",
        )
        _chart(fig_cm)

    if hpafl_data and "f1_per_class" in hpafl_data:
        st.subheader("Per-Class F1 Score")
        f1_data = hpafl_data["f1_per_class"]
        fig_f1 = px.bar(
            pd.DataFrame({"Class": list(f1_data.keys()), "F1": list(f1_data.values())}),
            x="Class", y="F1", color="Class",
            title="Per-Class F1 Score (HPAFL)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_f1.update_layout(template="plotly_white", showlegend=False)
        _chart(fig_f1)


# ---------------------------------------------------------------------------
# Page 3: Live Inference
# ---------------------------------------------------------------------------

elif page == "Live Inference":
    st.header("Live Inference — Dermoscopic Image Classification")
    st.warning(
        "**Disclaimer**: This tool is for research purposes only. "
        "Results are NOT a clinical diagnosis. Always consult a qualified dermatologist."
    )

    # Check if API is reachable before showing uploader
    api_ok = False
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3)
        api_ok = health.status_code == 200
    except Exception:
        pass

    if not api_ok:
        st.error(
            f"HPAFL API is not running on {API_BASE}. "
            "Start it in a separate terminal with:\n\n"
            "```bash\nuvicorn deployment.api:app --host 0.0.0.0 --port 8000\n```"
        )
    else:
        uploaded = st.file_uploader(
            "Upload a dermoscopic skin image (JPEG or PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(uploaded, caption="Uploaded image", use_container_width=True)

            with col2:
                with st.spinner("Classifying image..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/predict",
                            files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                            timeout=30,
                        )
                        result = response.json() if response.status_code == 200 else None
                        if result is None:
                            st.error(f"API error {response.status_code}: {response.text}")
                    except requests.ConnectionError:
                        st.error("Lost connection to API server.")
                        result = None

                if result:
                    st.subheader(f"Prediction: **{result['predicted_class']}**")
                    conf = result["confidence"]
                    st.progress(conf, text=f"Confidence: {conf:.1%}")

                    st.subheader("Top-3 Predictions")
                    df_top3 = pd.DataFrame(result["top3_predictions"])
                    if "confidence" in df_top3.columns:
                        df_top3["confidence"] = df_top3["confidence"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df_top3, use_container_width=True)

                    st.subheader("Clinical Note")
                    st.info(result.get("clinical_note", ""))

                    st.caption(f"Model version: {result.get('model_version', 'unknown')}")
                    st.caption(result.get("disclaimer", ""))
