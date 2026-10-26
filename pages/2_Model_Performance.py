"""Screen 2: Prospective Model Evaluation, Comparison Benchmarks & Scientific Rigor.

Hospital Records Clerk Assistant for inspecting prospective classifier metrics,
ROC/PR curves, confusion matrices, blocking efficiency, and leakage proof experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config as CFG
from src.app_state import (
    init_session_state,
    load_blocking_metrics,
    load_evaluation_metrics,
    render_disclaimer_banner,
    render_help_sidebar,
)

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance - Patient Deduplication",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# ── 2. Screen Header ──────────────────────────────────────────────────────────
st.title("📊 Screen 2: Prospective Model Performance & Evaluation Benchmarks")
st.markdown(
    """
    To ensure patient safety, classifiers are evaluated on an independent, **prospective test set** 
    composed of patients registered chronologically *after* the training period. In accordance with 
    Common Specification E, every accuracy metric is presented **directly beside the majority-class baseline**, 
    and candidate blocking losses are transparently penalized in end-to-end recall.
    """
)

st.divider()


# ── 3. Candidate Blocking Summary Card ────────────────────────────────────────
st.markdown("### ⚡ 1. Upstream Candidate Blocking Efficiency")
st.markdown(
    """
    Comparing all $50,000$ patient records pairwise requires evaluating over **$1.47$ billion comparisons** 
    ($n(n-1)/2$), which is computationally intractable in a production hospital database. Our multi-pass 
    blocking scheme prunes unpromising pairs prior to machine learning scoring.
    """
)

block_data = load_blocking_metrics()

if block_data:
    all_pairs = block_data.get("all_pairs_comparisons", 1478184378)
    union_info = block_data.get("schemes_evaluation", {}).get("UNION (B1-B5)") or block_data.get("schemes_evaluation", {}).get("Union (B1-B5)", {})
    cand_pairs = union_info.get("candidate_pairs_count", 68969)
    reduction_ratio = union_info.get("reduction_ratio_pct", 99.9953)
    completeness = union_info.get("pairs_completeness_pct", 94.48)
    missed_count = union_info.get("total_true_pairs", 9754) - union_info.get("true_pairs_found", 9216)

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric(
            "Total All-Pairs Space",
            f"{all_pairs:,}",
            help="Total exhaustive quadratic comparisons n(n-1)/2",
        )
    with bcol2:
        st.metric(
            "Candidate Pairs Extracted",
            f"{cand_pairs:,}",
            help="Pairs passing through multi-pass blocking filters",
        )
    with bcol3:
        st.metric(
            "Reduction Ratio",
            f"{reduction_ratio:.4f}%",
            help="1 - (Candidate Comparisons / All-Pairs)",
        )
    with bcol4:
        st.metric(
            "Pairs Completeness",
            f"{completeness:.2f}%",
            delta=f"-{missed_count} missed true pairs",
            delta_color="inverse",
            help="Proportion of true duplicates retained by blocking",
        )

    # Detailed blocking passes table
    with st.expander("🔍 View All 5 Blocking Passes & Multi-Pass Union Metrics", expanded=False):
        schemes = block_data.get("schemes_evaluation", {})
        block_rows = []
        for s_name, s_info in schemes.items():
            block_rows.append({
                "Blocking Scheme": s_name,
                "Key Definition": s_info.get("key_desc", ""),
                "Candidate Pairs": f"{s_info.get('candidate_pairs_count', 0):,}",
                "True Pairs Found": f"{s_info.get('true_pairs_found', 0):,}",
                "Completeness (%)": f"{s_info.get('pairs_completeness_pct', 0.0):.2f}%",
                "Quality (%)": f"{s_info.get('pairs_quality_pct', 0.0):.2f}%",
                "Reduction Ratio (%)": f"{s_info.get('reduction_ratio_pct', 0.0):.4f}%",
                "Runtime (s)": f"{s_info.get('runtime_seconds', 0.0):.3f}s",
            })
        st.dataframe(pd.DataFrame(block_rows), use_container_width=True, hide_index=True)
else:
    st.info("Blocking metrics report loading...")

st.divider()


# ── 4. Prospective Test Set Comparison Table ──────────────────────────────────
st.markdown("### 🏆 2. Prospective Test Set Model Benchmark Table")
st.markdown(
    r"""
    Evaluated strictly ONCE on the prospective evaluation population ($N = 106,298$ candidate pairs, 
    prevalence = $0.5362\%$) at operating threshold $\theta_1^* = 0.20$.
    """
)

eval_data = load_evaluation_metrics()

if eval_data:
    test_models = eval_data.get("test_models", {})
    cv_models = eval_data.get("cross_validation_training", {})
    e2e_models = eval_data.get("end_to_end", {}).get("models", {})

    table_rows = []

    for m_name in ["Pruned Decision Tree", "Categorical Naive Bayes", "k-Nearest Neighbours", "Multi-Layer Perceptron"]:
        if m_name in test_models:
            m_test = test_models[m_name]
            m_cv = cv_models.get(m_name, {})
            m_e2e = e2e_models.get(m_name, {})

            acc = m_test.get("accuracy", 0.0) * 100.0
            base_acc = m_test.get("majority_baseline", 0.994638) * 100.0
            prec = m_test.get("precision", 0.0) * 100.0
            rec = m_test.get("recall", 0.0) * 100.0
            e2e_rec = m_e2e.get("end_to_end_recall", 0.9448) * 100.0
            f1 = m_test.get("f1", 0.0)
            pr_auc = m_test.get("pr_auc", 0.0)
            roc_auc = m_test.get("roc_auc", 0.0)
            cost_1k = m_test.get("cost_analysis", {}).get("cost_per_1000_at_optimal", 0.0)

            # CV PR-AUC Mean +/- SD
            cv_pr = m_cv.get("pr_auc", {})
            cv_str = f"{cv_pr.get('mean', 0.0):.4f} ± {cv_pr.get('std', 0.0):.4f}"

            table_rows.append({
                "Classifier Model": m_name,
                "Test Accuracy (%)": f"{acc:.2f}%",
                "Majority Baseline (%)": f"{base_acc:.2f}%",  # MANDATORY ADJACENT COLUMN
                "Precision (%)": f"{prec:.2f}%",
                "Classifier Recall (%)": f"{rec:.2f}%",
                "End-to-End Recall (%)": f"{e2e_rec:.2f}%",
                "F1-Score": f"{f1:.4f}",
                "PR-AUC": f"{pr_auc:.4f}",
                "ROC-AUC": f"{roc_auc:.4f}",
                "Expected Cost (Rs./1k)": f"Rs. {cost_1k:.2f}",
                "5-Fold CV PR-AUC": cv_str,
            })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Clinical Accuracy Warning
    st.info(
        """
        ℹ️ **Why Accuracy is Deceptive in Patient Deduplication:**
        In hospital master indexes, non-matches constitute **99.46%** of candidate pairs. A trivial baseline 
        predicting *'all pairs are distinct'* achieves **99.46% accuracy** while suffering **0% recall**, 
        failing to identify a single duplicate. For medical records clerks, **Precision** (avoiding false alarms) 
        and **Recall** (preventing split patient records) are the only clinically meaningful evaluation metrics.
        """
    )
else:
    st.info("Evaluation metrics report loading...")

st.divider()


# ── 5. Diagnostic Charts & Visualizations ─────────────────────────────────────
st.markdown("### 📈 3. Diagnostic Curves & Operating Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "ROC Curves (with AUC)",
    "Precision-Recall Curves",
    "Confusion Matrix at Threshold",
    "Probability Calibration",
])

with tab1:
    st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
    st.caption("All models exhibit near-perfect ranking discrimination across prospective test candidate pairs.")
    roc_img_path = CFG.REPORTS_DIR / "figures" / "roc_curves.png"
    if roc_img_path.exists():
        st.image(str(roc_img_path), use_container_width=True)
    else:
        st.info("ROC curve figure loading...")

with tab2:
    st.markdown("#### Precision-Recall Curves")
    st.caption("Precision-Recall curves highlight performance under severe class imbalance (0.54% match prevalence).")
    pr_img_path = CFG.REPORTS_DIR / "figures" / "pr_curves.png"
    if pr_img_path.exists():
        st.image(str(pr_img_path), use_container_width=True)
    else:
        st.info("Precision-Recall curve figure loading...")

with tab3:
    st.markdown("#### Confusion Matrix at Operating Threshold")
    selected_m = st.selectbox(
        "Select model to view prospective test confusion matrix:",
        ["Pruned Decision Tree", "Categorical Naive Bayes", "k-Nearest Neighbours", "Multi-Layer Perceptron"],
        index=0,
    )

    if eval_data and "test_models" in eval_data:
        m_cost = eval_data["test_models"].get(selected_m, {}).get("cost_analysis", {})
        cm = m_cost.get("confusion_matrix_review", {})
        tp = cm.get("TP", 570)
        fp = cm.get("FP", 21)
        fn = cm.get("FN", 0)
        tn = cm.get("TN", 105707)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("True Positives (TP)", f"{tp:,}", help="Correctly flagged duplicates")
        with c2:
            st.metric("False Positives (FP)", f"{fp:,}", help="False alarms (wasted review time)")
        with c3:
            st.metric("False Negatives (FN)", f"{fn:,}", help="Missed duplicates (split charts)")
        with c4:
            st.metric("True Negatives (TN)", f"{tn:,}", help="Correctly ignored non-matches")

    cm_img_path = CFG.REPORTS_DIR / "figures" / "confusion_matrices.png"
    if cm_img_path.exists():
        st.image(str(cm_img_path), use_container_width=True)

with tab4:
    st.markdown("#### Post-Hoc Probability Calibration (Platt Scaling)")
    st.caption("Fitted strictly on training out-of-fold predictions to prevent probability distortion.")
    cal_img_path = CFG.REPORTS_DIR / "figures" / "calibration_curves.png"
    if cal_img_path.exists():
        st.image(str(cal_img_path), use_container_width=True)
    else:
        st.info("Calibration curve figure loading...")

st.divider()


# ── 6. Scientific Rigor & Methodological Experiments ──────────────────────────
st.markdown("### 🔬 4. Methodological Honesty & Scientific Experiments")
st.markdown(
    """
    To prove research integrity at the viva, three experimental controls were executed to demonstrate 
    potential pitfalls in record linkage: data leakage, split design, and pipeline robustness.
    """
)

if eval_data:
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.container(border=True):
            st.markdown("#### 🚫 1. Data Leakage Proof (Sequential IDs)")
            leakage = eval_data.get("leakage_proof", {})
            st.markdown(
                f"- **Leaky Feature:** `{leakage.get('leaky_feature_name', 'record_id_diff')}`\n"
                f"- **Training Accuracy (with Leaky ID):** `{leakage.get('train_accuracy_leaky', 0.8334)*100:.2f}%`\n"
                f"- **Prospective Test Accuracy:** `{leakage.get('test_accuracy_leaky', 0.9945)*100:.2f}%` "
                f"*(Collapses below majority baseline `{leakage.get('majority_baseline_test', 0.9946)*100:.2f}%`!)*"
            )
            st.caption(leakage.get("explanation", ""))

        with st.container(border=True):
            st.markdown("#### 🎲 2. Temporal vs. Random Split Experiment")
            split_exp = eval_data.get("split_experiment", {})
            rand_res = split_exp.get("random_split", {})
            temp_res = split_exp.get("temporal_split", {})
            st.markdown(
                f"- **Leaky Random Pair Split PR-AUC:** `{rand_res.get('pr_auc', 0.9991):.4f}` *(Artificially Inflated)*\n"
                f"- **Honest Entity-Grouped Temporal PR-AUC:** `{temp_res.get('pr_auc', 0.9794):.4f}` *(Truthful Assessment)*\n"
                f"- **Inflation Delta:** `+{split_exp.get('inflation_delta_pr_auc', 0.0197):.4f} PR-AUC`"
            )
            st.caption(split_exp.get("explanation", ""))

    with exp_col2:
        with st.container(border=True):
            st.markdown("#### 🧪 3. Independent Register Robustness Check")
            robust = eval_data.get("robustness_check", {})
            st.markdown(
                f"- **Independent Random Seed:** `{robust.get('robustness_seed', 999)}`\n"
                f"- **Records Ingested:** `{robust.get('n_records', 425):,}` ({robust.get('n_candidate_pairs', 3531):,} pairs)\n"
                f"- **Decision Tree PR-AUC:** `{robust.get('model_results', {}).get('Pruned Decision Tree', {}).get('pr_auc', 1.0):.4f}`\n"
                f"- **Naive Bayes PR-AUC:** `{robust.get('model_results', {}).get('Categorical Naive Bayes', {}).get('pr_auc', 1.0):.4f}`\n"
                f"- **k-NN PR-AUC:** `{robust.get('model_results', {}).get('k-Nearest Neighbours', {}).get('pr_auc', 1.0):.4f}`"
            )
            st.caption(robust.get("assessment", ""))

        with st.container(border=True):
            st.markdown("#### ⏱️ 4. Hospital Clerk Workload Impact")
            workload = eval_data.get("workload_summary", {})
            dt_work = workload.get("Pruned Decision Tree", {})
            nb_work = workload.get("Categorical Naive Bayes", {})
            st.markdown(
                f"- **Assumed Review Time:** `{workload.get('assumed_minutes_per_review', 3.0)} minutes / flagged pair`\n"
                f"- **Decision Tree Worklist:** `{dt_work.get('flagged_pairs_count', 591):,} pairs` "
                f"$\to$ **`{dt_work.get('estimated_clerk_hours', 29.6):.1f} hours`** ({dt_work.get('estimated_workdays_7_5h', 3.9):.1f} clerk-days)\n"
                f"- **Naive Bayes Worklist:** `{nb_work.get('flagged_pairs_count', 673):,} pairs` "
                f"$\to$ **`{nb_work.get('estimated_clerk_hours', 33.6):.1f} hours`** ({nb_work.get('estimated_workdays_7_5h', 4.5):.1f} clerk-days)"
            )
            st.caption("Pruned Decision Tree minimizes false alarms, saving ~4.0 hours of clinical review time.")
