"""Screen 5: Decision Policy, Threshold Tuning & Hospital Cost Matrix.

Enables hospital informatics officers and records managers to:
1. Select active classifier family across the application session.
2. Dynamically adjust operating decision threshold (with cost-optimal marker and reset button).
3. Customize the asymmetric clinical cost matrix (C_FP clerk review time vs C_FN split chart risk).
4. Simulate LIVE consequences via src/evaluate.py's `compute_cost_and_thresholds`:
   - Expected cost per 1,000 pairs
   - Precision and Recall
   - Clerk worklist queue size and estimated review hours
   - Confusion matrix
   - Accuracy given up compared to majority baseline and naive 0.50 threshold.
5. Persist all policy variables in st.session_state for Screens 1, 2, 3, 4, and 6.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import config as CFG
from src.app_state import (
    DEFAULT_COST_MATRIX,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_STRICT_THRESHOLD,
    get_test_predictions,
    init_session_state,
    render_disclaimer_banner,
    render_help_sidebar,
)
from src.evaluate import compute_cost_and_thresholds

st.set_page_config(
    page_title="Settings & Cost Matrix - Patient Deduplication",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# ── Title & Introduction ───────────────────────────────────────────────────────
st.title("⚙️ Screen 5: Decision Policy, Threshold Tuning & Hospital Cost Matrix")
st.markdown(
    "Configure hospital-specific decision policies, customize asymmetric misclassification costs, "
    "and observe **live simulated operational consequences** across the prospective test cohort."
)
st.divider()

# ── Model Families Catalogue ──────────────────────────────────────────────────
MODEL_OPTIONS = {
    "decision_tree": {
        "title": "Pruned Decision Tree (Recommended Primary Classifier)",
        "badge": "Primary / Explainable",
        "desc": "Fast rule-based tree optimized with cost-complexity pruning (ccp_alpha=0.0001). Provides transparent human-readable decision paths for clerks.",
    },
    "naive_bayes": {
        "title": "Categorical Naive Bayes",
        "badge": "Probabilistic / Fast",
        "desc": "Multi-category naive Bayes on supervised entropy discretization with Laplace smoothing (alpha=1.0). Decomposes evidence into log-likelihood ratios.",
    },
    "knn": {
        "title": "k-Nearest Neighbours (k-NN)",
        "badge": "Instance-Based / Geometric",
        "desc": "k=11 nearest neighbours with Euclidean distance weighting on standardized continuous metrics. Explains predictions via nearest historical training cases.",
    },
    "mlp": {
        "title": "Multi-Layer Perceptron (MLP)",
        "badge": "Deep Neural Network",
        "desc": "Nonlinear multi-layer perceptron (32, 16) with Adam optimizer and early stopping regularization.",
    },
}

current_model_key = st.session_state.get("selected_model", "decision_tree")
if current_model_key not in MODEL_OPTIONS:
    current_model_key = "decision_tree"

# ── Top Policy Controls: Model & Cost Matrix ──────────────────────────────────
col_model, col_costs = st.columns([1, 1], gap="medium")

with col_model:
    st.subheader("1. Active Classifier Family")
    st.caption("Select the machine learning model that powers single predictions, batch scoring, and merge reviews.")

    selected_model_idx = list(MODEL_OPTIONS.keys()).index(current_model_key)
    chosen_model = st.selectbox(
        "Active Classifier",
        options=list(MODEL_OPTIONS.keys()),
        index=selected_model_idx,
        format_func=lambda k: f"{MODEL_OPTIONS[k]['title']}",
        help="The active model is saved in session state and used across all screens in this browser session.",
    )

    if chosen_model != current_model_key:
        st.session_state["selected_model"] = chosen_model
        st.rerun()

    info = MODEL_OPTIONS[chosen_model]
    st.info(f"**{info['badge']}**: {info['desc']}")

with col_costs:
    st.subheader("2. Hospital Clinical Cost Matrix")
    st.caption("Define the financial and clinical risk values for pairwise misclassification.")

    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        cost_fp_val = st.number_input(
            "Cost of False Alarm (C_FP)",
            min_value=0.5,
            max_value=100.0,
            value=float(st.session_state.get("cost_fp", CFG.DEFAULT_COST_FP)),
            step=0.5,
            help="Cost incurred when distinct patients are flagged as duplicate, wasting clerk review time (units: PKR / pair).",
        )
        st.caption("💡 **Assumption:** Clerk spends 3–5 minutes manually reviewing one false alarm pair.")

    with cost_col2:
        cost_fn_val = st.number_input(
            "Cost of Missed Duplicate (C_FN)",
            min_value=1.0,
            max_value=500.0,
            value=float(st.session_state.get("cost_fn", CFG.DEFAULT_COST_FN)),
            step=1.0,
            help="Cost incurred when true duplicate patients are missed, leaving split medical charts (units: PKR / pair).",
        )
        st.caption("⚠️ **Assumption:** Split records risk repeated lab panels, missed allergies, and clinical divergence.")

    # Update session state with edited costs
    st.session_state["cost_fp"] = cost_fp_val
    st.session_state["cost_fn"] = cost_fn_val
    st.session_state["cost_matrix"]["cost_fp"] = cost_fp_val
    st.session_state["cost_matrix"]["cost_fn"] = cost_fn_val

    # Theoretical Bayesian Cost-Optimal Threshold
    # θ* = C_FP / (C_FP + C_FN)
    theoretical_optimal_theta = float(cost_fp_val / (cost_fp_val + cost_fn_val))
    st.session_state["cost_optimal_theta"] = theoretical_optimal_theta

st.divider()

# ── Threshold Tuning Section ──────────────────────────────────────────────────
st.subheader("3. Operating Decision Threshold (θ)")
st.caption(
    "Candidate pairs with duplicate probability $p \\ge \\theta$ are dispatched to the clerk review worklist. "
    "The Bayesian cost-optimal boundary is highlighted below."
)

thresh_col1, thresh_col2 = st.columns([3, 1], gap="medium")

with thresh_col1:
    current_threshold = float(st.session_state.get("threshold", theoretical_optimal_theta))
    # Ensure threshold within [0.01, 0.99]
    current_threshold = max(0.01, min(0.99, current_threshold))

    chosen_threshold = st.slider(
        "Operating Review Threshold (θ)",
        min_value=0.01,
        max_value=0.99,
        value=round(current_threshold, 2),
        step=0.01,
        help="Pairs with probability >= threshold are flagged for manual clerk review.",
    )
    st.session_state["threshold"] = chosen_threshold

    st.markdown(
        f"🎯 **Cost-Optimal Value for Current Matrix:** `θ* = {theoretical_optimal_theta:.4f}` "
        f"*(Formula: $\\frac{{C_{{\\text{{FP}}}}}}{{C_{{\\text{{FP}}}} + C_{{\\text{{FN}}}}}} = \\frac{{{cost_fp_val:.1f}}}{{{cost_fp_val:.1f} + {cost_fn_val:.1f}}}$)*"
    )

with thresh_col2:
    st.markdown("#### Threshold Actions")
    if st.button("🎯 Reset to Cost-Optimal", use_container_width=True, help="Set threshold exactly to theoretical cost-optimal θ*"):
        st.session_state["threshold"] = round(theoretical_optimal_theta, 4)
        st.rerun()

    if st.button("⚖️ Reset to Naive 0.50", use_container_width=True, help="Set threshold to balanced 0.50 baseline"):
        st.session_state["threshold"] = 0.50
        st.rerun()

st.divider()

# ── Live Consequence Simulation (compute_cost_and_thresholds) ──────────────────
st.subheader("4. LIVE Consequence Simulation (Prospective Test Cohort)")
st.caption(
    "All figures are recomputed in real time using `src/evaluate.py:compute_cost_and_thresholds()` "
    "across the prospective test set (106,298 candidate pairs, 570 true duplicates, natural prevalence 0.5362%)."
)

# Load test data and predictions
try:
    with st.spinner("Recomputing live operational metrics across prospective test set..."):
        y_true, p_pred = get_test_predictions(chosen_model)
        live_results = compute_cost_and_thresholds(
            y_true=y_true,
            p_pred=p_pred,
            cost_fp=cost_fp_val,
            cost_fn=cost_fn_val,
            operating_threshold=chosen_threshold,
        )
except Exception as e:
    st.error(f"Error computing live metrics: {e}")
    st.stop()

op_metrics = live_results["operating_metrics"]
naive_metrics = live_results["naive_metrics"]
maj_baseline = op_metrics["majority_baseline"]  # 0.994638

# Worklist metrics
flagged_count = op_metrics["flagged_count"]
clerk_minutes_per_pair = 3.0  # 3 minutes per review
est_clerk_hours = (flagged_count * clerk_minutes_per_pair) / 60.0
est_clerk_workdays = est_clerk_hours / 7.5  # 7.5 hour workday

# Key metric cards
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    cost_diff = op_metrics["expected_cost_1k"] - naive_metrics["expected_cost_1k"]
    st.metric(
        "Expected Cost / 1k Pairs",
        f"Rs. {op_metrics['expected_cost_1k']:.2f}",
        delta=f"{cost_diff:+.2f} vs θ=0.50",
        delta_color="inverse",
        help="Expected total misclassification cost per 1,000 candidate pair comparisons evaluated.",
    )

with m_col2:
    st.metric(
        "Classifier Recall",
        f"{op_metrics['recall'] * 100:.2f}%",
        delta=f"{(op_metrics['recall'] - naive_metrics['recall']) * 100:+.2f}% vs θ=0.50",
        help="Proportion of true duplicate pairs successfully identified and captured.",
    )

with m_col3:
    st.metric(
        "Classifier Precision",
        f"{op_metrics['precision'] * 100:.2f}%",
        delta=f"{(op_metrics['precision'] - naive_metrics['precision']) * 100:+.2f}% vs θ=0.50",
        help="Proportion of flagged candidate pairs that are genuine duplicates.",
    )

with m_col4:
    st.metric(
        "Worklist Queue Size",
        f"{flagged_count:,} pairs",
        help="Total number of candidate pairs sent to records clerks for manual review (TP + FP).",
    )

with m_col5:
    st.metric(
        "Est. Clerk Workload",
        f"{est_clerk_hours:.1f} hrs",
        delta=f"{est_clerk_workdays:.1f} workdays",
        delta_color="off",
        help="Estimated manual review time assuming 3.0 minutes per flagged candidate pair at 7.5 hours/day.",
    )

st.write("")

# ── Trade-off Analysis & Confusion Matrix ─────────────────────────────────────
sub_col1, sub_col2 = st.columns([1, 1], gap="medium")

with sub_col1:
    st.markdown("#### ⚖️ Clinical Accuracy Trade-off")
    acc_val = op_metrics["accuracy"]
    acc_given_up = live_results["accuracy_given_up"]

    # Mandatory baseline adjacency check!
    st.markdown(
        f"""
        | Metric | Operating Policy (θ = {chosen_threshold:.2f}) | Naive Threshold (θ = 0.50) | **Majority Baseline (Adjacent)** |
        |---|:---:|:---:|:---:|
        | **Overall Accuracy** | **{acc_val * 100:.2f}%** | {naive_metrics['accuracy'] * 100:.2f}% | **{maj_baseline * 100:.2f}%** |
        | **Expected Cost / 1k** | **Rs. {op_metrics['expected_cost_1k']:.2f}** | Rs. {naive_metrics['expected_cost_1k']:.2f} | Rs. {maj_baseline * cost_fn_val * 0.005362 * 1000 / 1000:.2f} |
        | **Recall (Duplicates)** | **{op_metrics['recall'] * 100:.2f}%** | {naive_metrics['recall'] * 100:.2f}% | 0.00% |
        | **Precision** | **{op_metrics['precision'] * 100:.2f}%** | {naive_metrics['precision'] * 100:.2f}% | 0.00% |
        """
    )

    if acc_given_up > 0:
        st.caption(
            f"ℹ️ **Accuracy Trade-off:** By operating at $\\theta = {chosen_threshold:.2f}$ instead of $\\theta = 0.50$, "
            f"the hospital accepts an apparent accuracy concession of **{acc_given_up * 100:.3f}%** "
            f"in exchange for capturing **{op_metrics['recall'] * 100:.1f}%** of duplicates and preventing split charts."
        )
    else:
        st.caption(
            f"ℹ️ **Accuracy Trade-off:** Operating at $\\theta = {chosen_threshold:.2f}$ achieves accuracy equal to or higher "
            f"than the naive 0.50 threshold while optimizing expected clinical costs."
        )

with sub_col2:
    st.markdown("#### 📊 Live Confusion Matrix (Counts)")
    tp = op_metrics["tp"]
    fp = op_metrics["fp"]
    fn = op_metrics["fn"]
    tn = op_metrics["tn"]

    cm_df = pd.DataFrame(
        [
            {"Actual Reality": "True Duplicate (Positive)", "Predicted Positive (Flagged)": f"TP = {tp:,} (Duplicates Merged)", "Predicted Negative (Ignored)": f"FN = {fn:,} (Split Charts Risk!)"},
            {"Actual Reality": "Distinct Records (Negative)", "Predicted Positive (Flagged)": f"FP = {fp:,} (Wasted Clerk Time)", "Predicted Negative (Ignored)": f"TN = {tn:,} (Correctly Ignored)"},
        ]
    ).set_index("Actual Reality")

    st.table(cm_df)

    st.markdown(
        f"""
        - **True Positives ($TP$):** `{tp:,}` true duplicate pairs captured for review/merge.
        - **False Positives ($FP$):** `{fp:,}` distinct pairs flagged (cost: Rs. `{fp * cost_fp_val:,.1f}`).
        - **False Negatives ($FN$):** `{fn:,}` missed duplicates (cost: Rs. `{fn * cost_fn_val:,.1f}`).
        - **True Negatives ($TN$):** `{tn:,}` distinct pairs correctly filtered out.
        """
    )

st.divider()

# ── Session State Persistence Confirmation ────────────────────────────────────
st.success(
    f"✅ **Session State Synced:** All screens are now configured with **{MODEL_OPTIONS[chosen_model]['title']}** "
    f"at review threshold **θ = {chosen_threshold:.2f}** (Cost Matrix: $C_{{FP}} = {cost_fp_val:.1f}, C_{{FN}} = {cost_fn_val:.1f}$).",
    icon="💾",
)

col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    st.page_link("pages/3_Single_Prediction.py", label="👉 Proceed to Screen 3: Single Pair Investigation", icon="🔍")
with col_nav2:
    st.page_link("pages/4_Batch_Scoring.py", label="👉 Proceed to Screen 4: Batch Register Scoring", icon="📋")
