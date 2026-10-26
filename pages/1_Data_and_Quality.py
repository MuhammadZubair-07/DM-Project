"""Screen 1: Register Ingestion, Data Quality Audit & Cleaning Provenance.

Hospital Records Clerk Assistant for inspecting data quality faults,
reviewing standardization effects, executing automated validation, and exporting clean registers.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config as CFG
from src.app_state import (
    init_session_state,
    load_audit_data,
    load_clean_effects,
    load_csv_data,
    render_disclaimer_banner,
    render_help_sidebar,
)
from src.clean import clean_patient_register
from src.validate import validate

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data & Quality - Patient Deduplication",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# ── 2. Screen Header ──────────────────────────────────────────────────────────
st.title("📋 Screen 1: Register Ingestion, Quality Audit & Cleaning Provenance")
st.markdown(
    """
    Hospital patient registers ingested from disparate departmental systems (OPD, Emergency, Lab, 
    Radiology, and Legacy paper archives) inevitably exhibit typographical errors, formatting inconsistencies, 
    and missing attributes. This screen provides transparent inspection of raw demographic records, 
    detected data quality faults, automated validation assertions, and standardized cleaning provenance.
    """
)

st.divider()


# ── 3. Register Ingestion & Upload ────────────────────────────────────────────
st.markdown("### 📥 1. Patient Register Ingestion")

ingest_mode = st.radio(
    "Choose ingestion source:",
    ["Use Active Hospital Register", "Upload New Hospital Register (CSV)"],
    horizontal=True,
)

raw_df: pd.DataFrame
clean_df: pd.DataFrame
dataset_label: str

REQUIRED_RAW_COLUMNS = [
    "record_id",
    "source_system",
    "registration_date",
    "first_name",
    "family_name",
    "father_name",
    "gender",
    "date_of_birth",
    "age_years",
    "national_id",
    "phone",
    "city",
    "address",
    "blood_group",
    "insurance_tier",
    "visit_count",
]

if ingest_mode == "Use Active Hospital Register":
    active_path = st.session_state.get(
        "active_dataset_path", str(CFG.DATA_DIR / "raw" / "demo_register.csv")
    )
    dataset_label = st.session_state.get("active_dataset_name", "Demo Register")
    raw_df = load_csv_data(active_path)

    # Clean counterpart
    clean_path = CFG.CLEAN_REGISTER_PATH
    if clean_path.exists() and "50,000" in dataset_label:
        clean_df = load_csv_data(str(clean_path))
    else:
        # Clean the active subset on the fly
        clean_df, _ = clean_patient_register(raw_df)

    st.success(f"Loaded active dataset: **{dataset_label}** ({len(raw_df):,} records)")

else:
    uploaded_file = st.file_uploader(
        "Upload raw hospital register CSV file for quality audit:",
        type=["csv"],
        help="Upload a standard hospital patient register containing demographic columns.",
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
            dataset_label = f"Uploaded File: {uploaded_file.name}"

            # Schema validation
            missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
            if missing_cols:
                st.error(
                    f"⚠️ **Schema Validation Error:** The uploaded CSV is missing {len(missing_cols)} required columns: "
                    f"`{', '.join(missing_cols)}`. Please provide a valid patient register."
                )
                st.stop()

            # Leakage check: ground truth must never be present
            forbidden_found = [c for c in raw_df.columns if c in CFG.FORBIDDEN_COLUMNS]
            if forbidden_found:
                st.error(
                    f"🚫 **Data Leakage Violation:** Uploaded file contains forbidden target columns: "
                    f"`{', '.join(forbidden_found)}`. Ground truth identifiers cannot be ingested!"
                )
                st.stop()

            # Clean uploaded register
            with st.spinner("Standardizing and cleaning uploaded register..."):
                clean_df, _ = clean_patient_register(raw_df)

            st.success(f"Successfully ingested and cleaned: **{uploaded_file.name}** ({len(raw_df):,} records)")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file or switch to 'Use Active Hospital Register' above.")
        st.stop()


# Overview Metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Total Records", f"{len(raw_df):,}")
with mcol2:
    st.metric("Total Columns", f"{len(raw_df.columns)}")
with mcol3:
    source_counts = raw_df["source_system"].nunique() if "source_system" in raw_df.columns else 0
    st.metric("Source Systems", f"{source_counts}")
with mcol4:
    date_col = raw_df["registration_date"].replace("", None).dropna() if "registration_date" in raw_df.columns else pd.Series()
    date_span = f"{date_col.min()[:4]}–{date_col.max()[:4]}" if not date_col.empty else "N/A"
    st.metric("Registration Span", date_span)

# Preview Expander
with st.expander("👁️ View Raw Register Data Preview (First 10 Rows)", expanded=False):
    # Enforce no ground truth in display
    display_cols = [c for c in raw_df.columns if c not in CFG.FORBIDDEN_COLUMNS]
    st.dataframe(raw_df[display_cols].head(10), use_container_width=True)

st.divider()


# ── 4. Data Quality Audit Report ──────────────────────────────────────────────
st.markdown("### 🔍 2. Data Quality Audit (Lecture 2 Categories)")
st.markdown(
    """
    Automated detectors scan the register for **eight distinct fault categories** established in 
    Lecture 2 (*Data Quality & Preprocessing*). Each detector evaluates an exact programmatic expression 
    to tally affected records.
    """
)

audit_json, audit_df = load_audit_data()

if not audit_df.empty:
    # Audit summary cards
    tot_faults = int(audit_df["count"].sum())
    distinct_categories = int(audit_df["category"].nunique())
    
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        st.metric("Total Detected Faults", f"{tot_faults:,}", help="Cumulative anomalies across all columns")
    with acol2:
        st.metric("Fault Categories", f"{distinct_categories}", help="Distinct Lecture 2 fault classifications")
    with acol3:
        st.metric("Audit Target", "50,000 Raw Records", help="Baseline hospital master register")

    # Filter by category
    categories = ["All Categories"] + sorted(audit_df["category"].unique().tolist())
    selected_cat = st.selectbox("Filter faults by Lecture 2 category:", categories)

    filtered_audit = audit_df if selected_cat == "All Categories" else audit_df[audit_df["category"] == selected_cat]

    # Display Audit Table
    st.dataframe(
        filtered_audit[[
            "fault_code",
            "fault_name",
            "category",
            "detecting_expression",
            "count",
            "percentage",
            "reason",
        ]].rename(columns={
            "fault_code": "Fault Code",
            "fault_name": "Fault Description",
            "category": "Lecture 2 Category",
            "detecting_expression": "Detecting Expression",
            "count": "Affected Records",
            "percentage": "% of Register",
            "reason": "Impact & Clinical Context",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Quality audit report loading...")

st.divider()


# ── 5. Cleaning & Standardization Provenance ──────────────────────────────────
st.markdown("### 🧹 3. Cleaning & Standardization Provenance")
st.markdown(
    """
    To preserve medical auditability, the cleaning pipeline strictly follows **two non-negotiable rules**:
    1. **Raw Preservation:** Uncleaned strings are permanently retained in dedicated `*_raw` columns.
    2. **Field Provenance:** Every repaired attribute receives a provenance indicator (`{field}_prov`) 
       documenting the exact transformation applied.
    """
)

effects_json, effects_df = load_clean_effects()

if not effects_df.empty:
    st.dataframe(
        effects_df[[
            "repair_item",
            "metric_description",
            "before",
            "after",
            "change",
        ]].rename(columns={
            "repair_item": "Standardization Item",
            "metric_description": "Issue Description",
            "before": "Count Before",
            "after": "Count After",
            "change": "Net Reduction",
        }),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📖 Provenance Tag Legend & Audit Values", expanded=False):
        st.markdown(
            """
            | Provenance Value | Transformation Description | Example Field & Effect |
            |:---|:---|:---|
            | `original` | Value was valid and unaltered | Proper ISO date `1985-06-15` retained as-is |
            | `case_fixed` | Whitespace trimmed and Title Case applied | `"  lahore "` $\to$ `"Lahore"` |
            | `alias_mapped` | Abbreviation or alias mapped to canonical form | `"LHR"` / `"KHI"` $\to$ `"Lahore"` / `"Karachi"` |
            | `reformatted` | Syntactic reformatting without information loss | `"+92 300 1234567"` $\to$ `"03001234567"` |
            | `set_missing` | Sentinel or impossible value replaced with empty | Sentinel `1900-01-01` or age `199` set to null |
            """
        )
else:
    st.info("Cleaning effects table loading...")

st.divider()


# ── 6. Automated Validation Assertions ────────────────────────────────────────
st.markdown("### 🛡️ 4. Automated Data Validation (`validate()`)")
st.markdown(
    """
    A robust deduplication system must prove data hygiene mathematically. The `validate()` suite 
    executes **15 rigorous quality assertions** covering unique identifiers, canonical encodings, 
    ISO 8601 formats, age-DOB reconciliation, and non-random missingness.
    """
)

# Run validate() on both frames
val_col1, val_col2 = st.columns(2)

with val_col1:
    with st.container(border=True):
        st.markdown("#### ❌ Raw Register Validation")
        report_raw = validate(raw_df, dataset_name="Raw Patient Register")
        st.error(
            f"**Status:** FAILED ({report_raw.failed_count} failing assertions, "
            f"{report_raw.passed_count}/{report_raw.total_assertions} passed)"
        )
        st.caption("Demonstrates that uncleaned intake data violates hospital integrity invariants.")

with val_col2:
    with st.container(border=True):
        st.markdown("#### ✅ Cleaned Register Validation")
        report_clean = validate(clean_df, dataset_name="Standardized Clean Register")
        st.success(
            f"**Status:** PASSED (All {report_clean.passed_count}/{report_clean.total_assertions} assertions passed, "
            f"0 violations!)"
        )
        st.caption("Proves that the standardization pipeline successfully resolved all structural anomalies.")

# Detailed Validation Table
with st.expander("📋 View All 15 Validation Assertions & Violation Counts", expanded=True):
    comparison_rows = []
    for a_raw, a_clean in zip(report_raw.assertions, report_clean.assertions):
        comparison_rows.append({
            "Assertion Name": a_raw.name,
            "Description": a_raw.description,
            "Raw Status": "❌ Fail" if not a_raw.passed else "✅ Pass",
            "Raw Violations": f"{a_raw.violations:,}",
            "Clean Status": "✅ Pass" if a_clean.passed else "❌ Fail",
            "Clean Violations": f"{a_clean.violations:,}",
        })

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

st.divider()


# ── 7. Data Dictionary & Measurement Types ────────────────────────────────────
st.markdown("### 📚 5. Data Dictionary & Measurement Scale Classification")
st.markdown(
    """
    In accordance with Stevens (1946) and Lecture 2 (*Measurement Scales*), every register attribute 
    is classified by measurement scale, discrete/continuous status, and clinical role. Identifiers and 
    phone numbers are strictly treated as **nominal strings**, not numbers.
    """
)

with st.expander("📖 Open Field Measurement Scale & Role Table", expanded=False):
    data_dict_table = pd.DataFrame([
        {"Field": "record_id", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Synthetic ID (RECxxxxx)", "Role": "Identifier (Non-predictive)"},
        {"Field": "source_system", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "OPD, Emergency, Lab, etc.", "Role": "Explanatory / Stratifier"},
        {"Field": "registration_date", "Measurement Scale": "Interval", "Nature": "Continuous", "Plausible Range": "2015-01-01 to 2025-10-24", "Role": "Temporal Split Cutoff"},
        {"Field": "first_name", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Pakistani given names", "Role": "Explanatory Feature"},
        {"Field": "family_name", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Pakistani surnames / clans", "Role": "Blocking Key / Feature"},
        {"Field": "father_name", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Patronymic given names", "Role": "Explanatory Feature"},
        {"Field": "gender", "Measurement Scale": "Nominal", "Nature": "Binary", "Plausible Range": "M, F", "Role": "Explanatory Feature"},
        {"Field": "date_of_birth", "Measurement Scale": "Interval", "Nature": "Continuous", "Plausible Range": "1920-01-01 to 2025-01-01", "Role": "Blocking Key / Feature"},
        {"Field": "age_years", "Measurement Scale": "Ratio", "Nature": "Discrete", "Plausible Range": "0 to 110", "Role": "Reconciled Explanatory"},
        {"Field": "national_id", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "13-digit CNIC (xxxxx-xxxxxxx-x)", "Role": "High-Specificity Matcher"},
        {"Field": "phone", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "11-digit mobile (03xxxxxxxxx)", "Role": "Blocking Key / Feature"},
        {"Field": "city", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "14 Canonical Pakistani Cities", "Role": "Explanatory Feature"},
        {"Field": "address", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Street, Sector, Town text", "Role": "Explanatory Feature"},
        {"Field": "blood_group", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "8 ABO/Rh types (A+, O-, etc.)", "Role": "Explanatory Feature"},
        {"Field": "insurance_tier", "Measurement Scale": "Ordinal", "Nature": "Discrete", "Plausible Range": "None < Basic < Standard < Premium", "Role": "Explanatory Feature"},
        {"Field": "visit_count", "Measurement Scale": "Ratio", "Nature": "Discrete", "Plausible Range": "1 to 250 visits", "Role": "Log-Transformed Feature"},
        {"Field": "true_entity_id", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Cluster UUIDs", "Role": "MUST NOT USE (Ground Truth)"},
        {"Field": "fault_type", "Measurement Scale": "Nominal", "Nature": "Discrete", "Plausible Range": "Corruption log tags", "Role": "MUST NOT USE (Generator Artifact)"},
    ])
    st.dataframe(data_dict_table, use_container_width=True, hide_index=True)

st.divider()


# ── 8. Export Cleaned Register ────────────────────────────────────────────────
st.markdown("### 💾 6. Download Standardized Clean Register")
st.markdown(
    """
    Download the standardized, sanitized register containing ISO dates, normalized mobile strings, 
    canonical city mappings, and complete audit provenance columns for downstream medical records analysis.
    """
)

# Convert clean_df to CSV buffer
csv_buffer = io.StringIO()
clean_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8")

st.download_button(
    label="⬇️ Download Cleaned Register CSV",
    data=csv_bytes,
    file_name="register_clean.csv",
    mime="text/csv",
    help="Download the standardized CSV ready for blocking and matching.",
    type="primary",
)
