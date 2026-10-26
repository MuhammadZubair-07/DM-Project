"""Streamlit Application Entry Point - Patient Record Deduplication Tool.

Hospital Records Clerk Assistant for similarity-based duplicate matching and reversible merges.
National Patient Master Index (NPMI) Project.
"""

from __future__ import annotations

import streamlit as st

from src.app_state import (
    init_session_state,
    render_disclaimer_banner,
    render_help_sidebar,
)

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Record Deduplication Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# Mandatory academic disclaimer banner on every page
render_disclaimer_banner()

# Shared help panel and sample data switcher
render_help_sidebar()


# ── 2. Hero Section ───────────────────────────────────────────────────────────
st.title("🏥 National Patient Master Index (NPMI)")
st.subheader("Hospital Records Clerk Assistant for Patient Deduplication & Reversible Linkage")

st.markdown(
    """
    Welcome to the **Patient Record Deduplication Tool**, an intelligent clinical portal designed specifically 
    for hospital medical records clerks. This system assists hospital staff in identifying, inspecting, and safely 
    merging split patient charts across legacy, outpatient, inpatient, and emergency intake systems.
    """
)

st.divider()


# ── 3. Six Dedicated Screen Cards ─────────────────────────────────────────────
st.markdown("### 📋 Application Modules & Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Ingestion & Quality Audit")
        st.markdown(
            "Inspect raw patient registers, examine detected data quality faults (Lecture 2 categories), "
            "review before/after cleaning provenance, and execute automated validation assertions."
        )
        st.page_link("pages/1_Data_and_Quality.py", label="Open Screen 1: Data & Quality", icon="📋")

    with st.container(border=True):
        st.markdown("#### 4. Batch Scoring & Worklist")
        st.markdown(
            "Upload patient registers, run high-performance candidate blocking, generate a ranked duplicate "
            "worklist filtered by confidence, and export prioritized review batches."
        )
        st.page_link("pages/4_Batch_Scoring.py", label="Open Screen 4: Batch Scoring", icon="📑")

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Prospective Model Performance")
        st.markdown(
            "Compare all model families against the majority-class baseline. Inspect ROC curves, precision-recall "
            "curves, calibration reliability, and leakage exclusion proofs."
        )
        st.page_link("pages/2_Model_Performance.py", label="Open Screen 2: Model Performance", icon="📊")

    with st.container(border=True):
        st.markdown("#### 5. Decision Policy & Cost Matrix")
        st.markdown(
            "Tune Bayesian operating thresholds, adjust false-alarm versus missed-duplicate hospital costs, "
            "and observe real-time trade-offs in clerk review workload and clinical safety."
        )
        st.page_link("pages/5_Settings_and_Cost_Matrix.py", label="Open Screen 5: Settings & Policy", icon="⚙️")

with col3:
    with st.container(border=True):
        st.markdown("#### 3. Single Prediction & Explanation")
        st.markdown(
            "Enter or populate any two patient charts. Receive calibrated duplicate probabilities, "
            "cost-derived confidence tiers, and plain-English reasons for and against a match."
        )
        st.page_link("pages/3_Single_Prediction.py", label="Open Screen 3: Single Prediction", icon="🔍")

    with st.container(border=True):
        st.markdown("#### 6. Reversible Merge Review")
        st.markdown(
            "Examine candidate duplicate pairs side-by-side with color-coded agreement flags. "
            "Select surviving golden record values and execute 100% reversible merges."
        )
        st.page_link("pages/6_Merge_Review.py", label="Open Screen 6: Merge Review", icon="🔀")

st.divider()


# ── 4. Core System Principles ─────────────────────────────────────────────────
st.markdown("### 🛡️ Core Operational & Clinical Safeguards")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("##### 🚫 Zero Technical Jargon")
    st.caption(
        "Clerks are never shown internal node IDs, Gini impurities, feature codes, or raw floats. "
        "Every decision split is translated into clear clinical English sentences."
    )

with info_col2:
    st.markdown("##### ⚖️ Cost-Derived Operating Bands")
    st.caption(
        "Decision thresholds are derived directly from hospital misclassification costs "
        r"($\theta^* = C_{\text{FP}} / (C_{\text{FP}} + C_{\text{FN}})$), establishing rigorous Review and Auto-Merge tiers."
    )

with info_col3:
    st.markdown("##### ↩️ Fully Reversible Merges")
    st.caption(
        "The system never alters records irrevocably. Every merged chart retains full field-level "
        "provenance and can be restored with a single click from the audit log."
    )
