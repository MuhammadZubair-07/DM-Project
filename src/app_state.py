"""Shared Application State, Session Store, and Utility UI Components.

Provides:
1. init_session_state(): Centralized st.session_state initialization for models,
   thresholds, cost matrices, and active datasets.
2. render_disclaimer_banner(): Mandatory warning banner rendered on every screen.
3. render_help_sidebar(): Clerk guidance panel, sample data switchers, and system status.
4. Cached data accessors with @st.cache_data for instant, responsive page transitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import config as CFG
from src.models import Matcher

# ── 1. Centralized Session State Initialization ───────────────────────────────

DEFAULT_COST_MATRIX = {
    "cost_fp": CFG.DEFAULT_COST_FP,         # Rs. 5.0 (clerk review time)
    "cost_fn": CFG.DEFAULT_COST_FN,         # Rs. 20.0 (split medical history risk)
    "cost_fp_strict": CFG.STRICT_COST_FP,   # Rs. 50.0 (harm of wrong patient merge)
    "cost_fn_strict": CFG.STRICT_COST_FN,   # Rs. 10.0 (review deferral)
}

# Derived Bayesian cost-optimal thresholds
DEFAULT_REVIEW_THRESHOLD: float = DEFAULT_COST_MATRIX["cost_fp"] / (
    DEFAULT_COST_MATRIX["cost_fp"] + DEFAULT_COST_MATRIX["cost_fn"]
)  # 0.20

DEFAULT_STRICT_THRESHOLD: float = DEFAULT_COST_MATRIX["cost_fp_strict"] / (
    DEFAULT_COST_MATRIX["cost_fp_strict"] + DEFAULT_COST_MATRIX["cost_fn_strict"]
)  # 0.8333


def init_session_state() -> None:
    """Initializes default application state variables if not already present."""
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "decision_tree"

    if "threshold" not in st.session_state:
        st.session_state["threshold"] = DEFAULT_REVIEW_THRESHOLD

    if "threshold_strict" not in st.session_state:
        st.session_state["threshold_strict"] = DEFAULT_STRICT_THRESHOLD

    if "cost_matrix" not in st.session_state:
        st.session_state["cost_matrix"] = dict(DEFAULT_COST_MATRIX)

    # Active Register File
    demo_path = CFG.DATA_DIR / "raw" / "demo_register.csv"
    if "active_dataset_name" not in st.session_state:
        st.session_state["active_dataset_name"] = "Demo Register (~5,000 records)"
        st.session_state["active_dataset_path"] = str(demo_path)

    if "cost_fp" not in st.session_state:
        st.session_state["cost_fp"] = float(DEFAULT_COST_MATRIX["cost_fp"])

    if "cost_fn" not in st.session_state:
        st.session_state["cost_fn"] = float(DEFAULT_COST_MATRIX["cost_fn"])

    if "batch_scored_results" not in st.session_state:
        st.session_state["batch_scored_results"] = None

    # Reversible Merge History
    if "merge_history" not in st.session_state:
        st.session_state["merge_history"] = []

    if "undo_stack" not in st.session_state:
        st.session_state["undo_stack"] = []


# ── 2. Mandatory Disclaimer Banner ────────────────────────────────────────────

def render_disclaimer_banner() -> None:
    """Renders the mandatory academic disclaimer banner required on EVERY page."""
    st.warning(
        "⚠️ **Student project. Not fit for clinical, financial or legal use. All data is synthetic.**"
    )


# ── 3. Shared Help Sidebar & Sample Data Selectors ─────────────────────────────

def render_help_sidebar() -> None:
    """Renders a consistent, friendly clerk guidance panel and sample data switcher."""
    with st.sidebar:
        st.markdown("### 🏥 Hospital Records Portal")
        st.caption("National Patient Master Index (NPMI) Deduplication Assistant")
        st.divider()

        # How to Use Help Panel
        with st.expander("ℹ️ How to Use This Portal", expanded=False):
            st.markdown(
                """
                **Clerk Workflow:**
                1. **Screen 1 (Data & Quality):** Inspect register quality, review detected faults, and verify standardization rules.
                2. **Screen 2 (Model Performance):** Compare prospective test accuracy against the majority baseline, examine ROC curves, and review leakage proofs.
                3. **Screen 3 (Single Prediction):** Compare two patient charts side-by-side with plain-English reasons for and against duplicate status.
                4. **Screen 4 (Batch Scoring):** Ingest raw patient registers and browse the ranked duplicate worklist.
                5. **Screen 5 (Settings):** Tune review thresholds and customize hospital misclassification costs.
                6. **Screen 6 (Merge Review):** Review potential duplicates, select field survivorship values, and execute 1-click reversible merges.
                """
            )

        # Sample Data Loaders
        st.markdown("#### 📂 Sample Data Selector")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Demo (~5k)", use_container_width=True, help="Load 5,000-record demo register"):
                st.session_state["active_dataset_name"] = "Demo Register (~5,000 records)"
                st.session_state["active_dataset_path"] = str(CFG.DATA_DIR / "raw" / "demo_register.csv")
                st.rerun()

        with col2:
            if st.button("Sample (~2k)", use_container_width=True, help="Load 2,000-record sample batch"):
                st.session_state["active_dataset_name"] = "Sample Batch (~2,000 records)"
                st.session_state["active_dataset_path"] = str(CFG.DATA_DIR / "raw" / "sample_batch.csv")
                st.rerun()

        if st.button("Full Register (~50k)", use_container_width=True, help="Load full 50,000-record register"):
            st.session_state["active_dataset_name"] = "Full Register (~50,000 records)"
            st.session_state["active_dataset_path"] = str(CFG.RAW_REGISTER_PATH)
            st.rerun()

        st.caption(f"**Active:** `{st.session_state.get('active_dataset_name', 'Default')}`")
        st.divider()

        # Active Model & Threshold Status Card
        model_display = {
            "decision_tree": "Pruned Decision Tree (Primary)",
            "naive_bayes": "Categorical Naive Bayes",
            "knn": "k-Nearest Neighbours",
            "mlp": "Multi-Layer Perceptron",
        }.get(st.session_state.get("selected_model", "decision_tree"), "Decision Tree")

        st.markdown("#### ⚙️ Active Decision Policy")
        st.markdown(f"- **Classifier:** `{model_display}`")
        st.markdown(f"- **Review Threshold ($\theta_1^*$):** `{st.session_state.get('threshold', 0.20):.2f}`")
        st.markdown(f"- **Auto-Merge Threshold ($\theta_2^*$):** `{st.session_state.get('threshold_strict', 0.8333):.2f}`")
        st.caption("Adjust policy parameters on **Screen 5 (Settings)**.")


# ── 4. Cached Data Loaders (Fast Responsive Transitions) ──────────────────────

@st.cache_data(show_spinner=False)
def load_csv_data(file_path: str) -> pd.DataFrame:
    """Caches CSV loading for fast rendering."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


@st.cache_data(show_spinner=False)
def load_json_report(file_path: str) -> dict[str, Any]:
    """Caches JSON report loading."""
    p = Path(file_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_audit_data() -> tuple[dict[str, Any], pd.DataFrame]:
    """Loads data quality audit metrics and CSV report."""
    audit_json = load_json_report(str(CFG.REPORTS_DIR / "audit.json"))
    csv_path = CFG.REPORTS_DIR / "audit.csv"
    audit_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return audit_json, audit_df


@st.cache_data(show_spinner=False)
def load_clean_effects() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Loads cleaning repair before/after effects table."""
    effects_json = load_json_report(str(CFG.CLEAN_EFFECTS_PATH))
    csv_path = CFG.CLEAN_EFFECTS_CSV_PATH
    effects_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return effects_json if isinstance(effects_json, list) else [], effects_df


@st.cache_data(show_spinner=False)
def load_blocking_metrics() -> dict[str, Any]:
    """Loads candidate blocking evaluation metrics."""
    return load_json_report(str(CFG.REPORTS_DIR / "blocking.json"))


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> dict[str, Any]:
    """Loads prospective test set evaluation metrics and scientific experiments."""
    return load_json_report(str(CFG.REPORTS_DIR / "evaluation.json"))


@st.cache_data(show_spinner=False)
def load_models_metrics() -> dict[str, Any]:
    """Loads hyper-parameter tuning and cross-validation metrics."""
    return load_json_report(str(CFG.REPORTS_DIR / "models.json"))


@st.cache_data(show_spinner=False)
def load_markdown_doc(file_path: str) -> str:
    """Reads markdown documentation files."""
    p = Path(file_path)
    if not p.exists():
        return "*Documentation file not found.*"
    return p.read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def get_matcher(model_name: str) -> Matcher:
    """Loads and caches a fitted Matcher instance."""
    clean_name = model_name.lower().replace(" ", "_")
    model_path = CFG.MODELS_DIR / f"{clean_name}.joblib"
    if not model_path.exists():
        model_path = CFG.MODELS_DIR / "decision_tree.joblib"
    return Matcher.load(model_path)


@st.cache_data(show_spinner=False)
def get_test_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads the prospective test partition and generates cached probability predictions.
    
    Returns:
        (y_true, p_pred) where y_true is 1D array of binary labels, and p_pred is 1D array of probabilities.
    """
    test_parquet = CFG.DATA_DIR / "pairs" / "test_features.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"Test feature file not found: {test_parquet}")
    df_test = pd.read_parquet(test_parquet)
    y_true = df_test["label"].to_numpy(dtype=int)
    matcher = get_matcher(model_name)
    p_pred = matcher.predict_proba(df_test)[:, 1]
    return y_true, p_pred

