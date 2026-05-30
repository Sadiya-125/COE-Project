"""Streamlit monitoring and inference dashboard for HPAFL.

Three pages:
    1. Training Monitor — per-hospital accuracy, weight evolution, privacy budget.
    2. Model Evaluation — baseline comparison, confusion matrix, F1 per class.
    3. Live Inference — upload an image and get a classification with clinical note.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from config import CLASS_FULL_NAMES, CLASS_NAMES, HAPFLConfig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HPAFL Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = HAPFLConfig()
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
    """Load adaptive_weights.csv if it exists."""
    p = RESULTS_DIR / "adaptive_weights.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _load_privacy_csv() -> Optional[pd.DataFrame]:
    """Load privacy_budget.csv if it exists."""
    p = RESULTS_DIR / "privacy_budget.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _load_results(fname: str) -> Optional[Dict]:
    """Load a results JSON file."""
    p = RESULTS_DIR / fname
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Page 1: Training Monitor
# ---------------------------------------------------------------------------

if page == "Training Monitor":
    st.header("Federated Training Progress")

    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()

    df_weights = _load_weights_csv()

    # ---- Metric cards ----
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

    # ---- Accuracy per round ----
    if df_weights is not None and not df_weights.empty:
        st.subheader("Accuracy per Hospital per FL Round")
        fig_acc = px.line(
            df_weights,
            x="round",
            y="accuracy",
            color="hospital_id",
            markers=True,
            title="Validation Accuracy per Hospital",
            labels={"round": "FL Round", "accuracy": "Accuracy", "hospital_id": "Hospital"},
        )
        fig_acc.update_layout(template="plotly_white")
        st.plotly_chart(fig_acc, use_container_width=True)

    # ---- Adaptive weights ----
    if df_weights is not None and not df_weights.empty:
        st.subheader("Adaptive Aggregation Weight Evolution")
        fig_w = px.bar(
            df_weights,
            x="round",
            y="weight",
            color="hospital_id",
            barmode="stack",
            title="Normalised Adaptive Weights per Round",
            labels={"round": "FL Round", "weight": "Weight", "hospital_id": "Hospital"},
        )
        fig_w.update_layout(template="plotly_white")
        st.plotly_chart(fig_w, use_container_width=True)

    # ---- Privacy budget ----
    df_privacy = _load_privacy_csv()
    if df_privacy is not None and not df_privacy.empty:
        st.subheader("Privacy Budget Consumption (Cumulative ε)")
        fig_priv = px.line(
            df_privacy,
            x="round",
            y="epsilon",
            color="hospital_id",
            markers=True,
            title="Cumulative Privacy Budget (ε) per Hospital",
            labels={"round": "FL Round", "epsilon": "Cumulative ε", "hospital_id": "Hospital"},
        )
        fig_priv.add_hline(
            y=cfg.target_epsilon,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Budget limit ε={cfg.target_epsilon}",
        )
        fig_priv.update_layout(template="plotly_white")
        st.plotly_chart(fig_priv, use_container_width=True)


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
            rows.append(
                {
                    "System": sname,
                    "Accuracy": f"{data.get('accuracy', 0):.4f}",
                    "F1-Macro": f"{data.get('f1_macro', 0):.4f}",
                    "ROC-AUC": f"{data.get('roc_auc_ovr', 0):.4f}",
                    "Precision": f"{data.get('precision_macro', 0):.4f}",
                    "Recall": f"{data.get('recall_macro', 0):.4f}",
                    "ε (privacy)": f"{data.get('epsilon', float('inf')):.2f}",
                }
            )

    if rows:
        st.subheader("Baseline Comparison Table")
        df_cmp = pd.DataFrame(rows)
        st.dataframe(
            df_cmp.style.highlight_max(
                subset=["Accuracy", "F1-Macro", "ROC-AUC"],
                color="lightgreen",
            ),
            use_container_width=True,
        )
    else:
        st.info("No results files found. Run training scripts first.")

    # Confusion matrix
    hpafl_data = _load_results("hpafl.json")
    if hpafl_data and "confusion_matrix" in hpafl_data:
        st.subheader("HPAFL Confusion Matrix")
        cm = hpafl_data["confusion_matrix"]
        import numpy as np
        cm_arr = np.array(cm, dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_arr / row_sums, 0.0)
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_norm,
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                colorscale="Blues",
                text=[[str(int(cm_arr[r][c])) for c in range(len(CLASS_NAMES))]
                      for r in range(len(CLASS_NAMES))],
                texttemplate="%{text}",
            )
        )
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="True",
            title="Confusion Matrix (row-normalised)",
            template="plotly_white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class F1
    if hpafl_data and "f1_per_class" in hpafl_data:
        st.subheader("Per-Class F1 Score")
        f1_data = hpafl_data["f1_per_class"]
        df_f1 = pd.DataFrame(
            {"Class": list(f1_data.keys()), "F1": list(f1_data.values())}
        )
        fig_f1 = px.bar(
            df_f1,
            x="Class",
            y="F1",
            color="Class",
            title="Per-Class F1 Score (HPAFL)",
            labels={"F1": "F1 Score"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_f1.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: Live Inference
# ---------------------------------------------------------------------------

elif page == "Live Inference":
    st.header("Live Inference — Dermoscopic Image Classification")
    st.warning(
        "⚠️ **Disclaimer**: This tool is for research purposes only. "
        "Results are NOT a clinical diagnosis. Always consult a qualified dermatologist."
    )

    uploaded = st.file_uploader(
        "Upload a dermoscopic skin image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        col1, col2 = st.columns([1, 2])

        # Preview
        with col1:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)

        # Call API
        with col2:
            with st.spinner("Classifying image…"):
                try:
                    response = requests.post(
                        f"{API_BASE}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                        result = None
                except requests.ConnectionError:
                    st.error(
                        "Cannot connect to HPAFL API. "
                        "Start it with: `uvicorn deployment.api:app --port 8000`"
                    )
                    result = None

            if result:
                # Top prediction
                st.subheader(f"Prediction: **{result['predicted_class']}**")
                conf = result["confidence"]
                st.progress(conf, text=f"Confidence: {conf:.1%}")

                # Top-3 table
                st.subheader("Top-3 Predictions")
                top3 = result["top3_predictions"]
                df_top3 = pd.DataFrame(top3)
                if "confidence" in df_top3.columns:
                    df_top3["confidence"] = df_top3["confidence"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_top3, use_container_width=True)

                # Clinical note
                st.subheader("Clinical Note")
                st.info(result.get("clinical_note", "No additional information available."))

                # Metadata
                st.caption(f"Model version: {result.get('model_version', 'unknown')}")
                st.caption(result.get("disclaimer", ""))
