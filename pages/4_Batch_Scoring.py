"""Screen 4: Batch Deduplication & Ranked Worklist.

Enables hospital records departments and audit clerks to:
1. Upload a full patient register CSV (or load the sample batch register).
2. Validate required columns and strip ground truth / synthetic metadata (leakage guard).
3. Execute the full end-to-end pipeline with an interactive progress bar:
   - Data cleaning and standardization
   - 5-pass candidate blocking (B1-B5)
   - Pair comparison feature extraction
   - Scoring with the active classifier chosen in Screen 5.
4. View blocking reduction statistics computed specifically for THIS file.
5. Inspect a ranked worklist prioritized by duplicate probability, showing confidence tiers,
   names, DOB, source systems, cities, and one-line plain-English reasons.
6. Apply interactive filters (confidence band, source system, city, minimum probability).
7. Download full scored results as CSV.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from rapidfuzz.distance import JaroWinkler

from src import config as CFG
from src.app_state import (
    get_matcher,
    init_session_state,
    render_disclaimer_banner,
    render_help_sidebar,
)
from src.blocking import (
    pass_b1_soundex_dob_year,
    pass_b2_fn3_dob_year_month,
    pass_b3_phone_last7,
    pass_b4_national_id_exact,
    pass_b5_sorted_neighbourhood,
)
from src.clean import clean_patient_register
from src.features import extract_features_for_pair_dataframe
from src.merge_store import MergeStore

st.set_page_config(
    page_title="Batch Scoring - Patient Deduplication",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# ── Title & Introduction ───────────────────────────────────────────────────────
st.title("📋 Screen 4: Batch Register Deduplication & Ranked Worklist")
st.markdown(
    "Ingest multi-source hospital registers, execute candidate blocking, score candidate pairs "
    "with the active classifier, and review a **prioritized duplicate worklist** with plain-English reasons."
)

# Active settings indicator
active_model = st.session_state.get("selected_model", "decision_tree")
active_thresh = float(st.session_state.get("threshold", 0.20))
active_thresh_strict = float(st.session_state.get("threshold_strict", 0.8333))

model_display = {
    "decision_tree": "Pruned Decision Tree",
    "naive_bayes": "Categorical Naive Bayes",
    "knn": "k-Nearest Neighbours",
    "mlp": "Multi-Layer Perceptron",
}.get(active_model, active_model)

st.caption(
    f"Active Classifier: **{model_display}** | "
    f"Review Threshold: **θ = {active_thresh:.2f}** | "
    f"Auto-Merge Threshold: **θ_strict = {active_thresh_strict:.2f}** "
    f"([Tune in Screen 5](pages/5_Settings_and_Cost_Matrix.py))"
)
st.divider()

# ── File Upload & Ingestion Controls ──────────────────────────────────────────
st.subheader("1. Patient Register Ingestion")

st.markdown(
    """
    > 💡 **Recommended File Size:** Up to **10,000 patient records (~5 MB)** for real-time interactive browser execution. 
    > Files with more than 10,000 records are supported and processed with block-size caps to guarantee browser responsiveness.
    """
)

col_up, col_sample = st.columns([2, 1], gap="medium")

uploaded_file = None
with col_up:
    uploaded_file = st.file_uploader(
        "Upload Raw Register CSV",
        type=["csv"],
        help="Upload a CSV patient register containing demographic columns (first_name, family_name, DOB, etc.).",
    )

with col_sample:
    st.markdown("#### Quick-Load Sample")
    sample_btn = st.button("📁 Load `sample_batch.csv` (1,801 records)", use_container_width=True)
    demo_btn = st.button("📁 Load `demo_register.csv` (5,000 records)", use_container_width=True)

# Determine data source
df_raw = None
source_label = ""

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
        source_label = f"Uploaded File: `{uploaded_file.name}` ({len(df_raw):,} records)"
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
elif sample_btn or st.session_state.get("batch_active_file") == "sample_batch":
    sample_path = CFG.DATA_DIR / "raw" / "sample_batch.csv"
    if sample_path.exists():
        df_raw = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        source_label = f"Sample Batch Register: `sample_batch.csv` ({len(df_raw):,} records)"
        st.session_state["batch_active_file"] = "sample_batch"
elif demo_btn or st.session_state.get("batch_active_file") == "demo_register":
    demo_path = CFG.DATA_DIR / "raw" / "demo_register.csv"
    if demo_path.exists():
        df_raw = pd.read_csv(demo_path, dtype=str, keep_default_na=False)
        source_label = f"Demo Register: `demo_register.csv` ({len(df_raw):,} records)"
        st.session_state["batch_active_file"] = "demo_register"
elif "batch_raw_df" in st.session_state:
    df_raw = st.session_state["batch_raw_df"]
    source_label = st.session_state.get("batch_source_label", "Active Cached Register")
else:
    # Default to sample_batch for initial presentation
    sample_path = CFG.DATA_DIR / "raw" / "sample_batch.csv"
    if sample_path.exists():
        df_raw = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        source_label = f"Sample Batch Register: `sample_batch.csv` ({len(df_raw):,} records)"
        st.session_state["batch_active_file"] = "sample_batch"

if df_raw is None or len(df_raw) == 0:
    st.info("Please upload a patient register CSV or click **Load sample_batch.csv** above to begin.")
    st.stop()

# Cache raw dataframe in session state
st.session_state["batch_raw_df"] = df_raw
st.session_state["batch_source_label"] = source_label

st.success(f"**Loaded:** {source_label}")

# ── Validation & Leakage Guard ────────────────────────────────────────────────
REQUIRED_COLS = ["first_name", "family_name", "gender", "date_of_birth"]
missing_req = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_req:
    st.error(f"❌ Uploaded CSV is missing critical demographic columns: `{missing_req}`. Required: `{REQUIRED_COLS}`.")
    st.stop()

# Ensure record_id exists
if "record_id" not in df_raw.columns:
    df_raw["record_id"] = [f"REC_{i+1:05d}" for i in range(len(df_raw))]

# Strip forbidden ground truth columns
leaked = [c for c in df_raw.columns if c in CFG.FORBIDDEN_COLUMNS]
if leaked:
    st.warning(f"⚠️ **Leakage Guard:** Stripped forbidden ground-truth / evaluation columns from dataset: `{leaked}`.")
    df_raw = df_raw.drop(columns=leaked)

st.divider()

# ── Pipeline Execution Section ────────────────────────────────────────────────
st.subheader("2. Candidate Blocking & Batch Scoring Pipeline")

run_scoring_col, _ = st.columns([1, 2])
with run_scoring_col:
    start_scoring = st.button("⚡ Run Deduplication Pipeline", type="primary", use_container_width=True)

# Check if we already have scored results for this file and model
has_cached_results = (
    st.session_state.get("batch_scored_results") is not None
    and st.session_state.get("batch_scored_model") == active_model
    and st.session_state.get("batch_scored_file") == source_label
)

if start_scoring or not has_cached_results:
    progress_bar = st.progress(0, text="Initializing clinical deduplication pipeline...")
    status_msg = st.empty()
    t_start = time.time()

    # Step 1: Cleaning
    status_msg.info("Step 1/4: Cleaning and standardizing patient records (Lecture 2)...")
    progress_bar.progress(20, text="Standardizing dates, names, phones, and sentinels...")
    df_clean, _ = clean_patient_register(df_raw)

    # Filter to model allowlist columns to prevent unallowlisted column warnings
    model_cols = [c for c in CFG.CLEAN_MODEL_ALLOWLIST if c in df_clean.columns]
    df_clean_model = df_clean[model_cols].copy()

    # Step 2: Multi-Pass Candidate Blocking
    status_msg.info("Step 2/4: Indexing multi-pass candidate blocking passes (B1–B5)...")
    progress_bar.progress(45, text="Building candidate pair index across 5 blocking passes...")

    t_block_start = time.time()
    p1, _, _, _ = pass_b1_soundex_dob_year(df_clean_model)
    p2, _, _, _ = pass_b2_fn3_dob_year_month(df_clean_model)
    p3, _, _, _ = pass_b3_phone_last7(df_clean_model)
    p4, _, _, _ = pass_b4_national_id_exact(df_clean_model)
    p5, _ = pass_b5_sorted_neighbourhood(df_clean_model, window_size=5)
    union_pairs = list(p1 | p2 | p3 | p4 | p5)
    t_block = time.time() - t_block_start

    n_records = len(df_clean_model)
    all_pairs_count = n_records * (n_records - 1) // 2
    n_candidates = len(union_pairs)
    reduction_ratio = 1.0 - (n_candidates / all_pairs_count) if all_pairs_count > 0 else 0.0

    st.session_state["batch_blocking_stats"] = {
        "n_records": n_records,
        "all_pairs_count": all_pairs_count,
        "n_candidates": n_candidates,
        "reduction_ratio": reduction_ratio,
        "blocking_time": t_block,
    }

    # Step 3: Feature Extraction
    status_msg.info(f"Step 3/4: Extracting 58 comparison features for {n_candidates:,} candidate pairs...")
    progress_bar.progress(70, text=f"Computing pairwise string distances and agreement features for {n_candidates:,} pairs...")

    if n_candidates == 0:
        st.warning("No candidate pairs generated by blocking passes.")
        st.stop()

    pairs_df = pd.DataFrame(union_pairs, columns=["record_id_1", "record_id_2"])
    feats_df = extract_features_for_pair_dataframe(pairs_df, df_clean_model)

    # Step 4: Model Scoring & Reason Generation
    status_msg.info(f"Step 4/4: Scoring candidate pairs with {model_display}...")
    progress_bar.progress(90, text="Applying trained machine learning pipeline and Platt calibrator...")

    matcher = get_matcher(active_model)
    probs = matcher.predict_proba(feats_df)[:, 1]

    # Quick Demographic Lookup Map
    recs_dict = df_clean.set_index("record_id").to_dict(orient="index")

    # Fast One-Line Reason Synthesizer
    reasons_list: list[str] = []
    r1_list = pairs_df["record_id_1"].tolist()
    r2_list = pairs_df["record_id_2"].tolist()

    for r1, r2, p in zip(r1_list, r2_list, probs):
        row1 = recs_dict.get(r1, {})
        row2 = recs_dict.get(r2, {})

        if p >= active_thresh:
            cues: list[str] = []
            # National ID
            nid1, nid2 = str(row1.get("national_id", "")), str(row2.get("national_id", ""))
            if nid1 and nid2 and nid1 == nid2:
                cues.append("Identical CNIC")

            # Phone
            ph1, ph2 = str(row1.get("phone", "")), str(row2.get("phone", ""))
            if ph1 and ph2 and ph1 == ph2:
                cues.append("Identical phone")
            elif ph1 and ph2 and ph1[-7:] == ph2[-7:]:
                cues.append("Matching subscriber phone digits")

            # DOB
            dob1, dob2 = str(row1.get("date_of_birth", "")), str(row2.get("date_of_birth", ""))
            if dob1 and dob2 and dob1 == dob2:
                cues.append("Exact same DOB")

            # First Name
            fn1, fn2 = str(row1.get("first_name", "")), str(row2.get("first_name", ""))
            if fn1 and fn2:
                if fn1.lower() == fn2.lower():
                    cues.append("Identical first name")
                elif JaroWinkler.similarity(fn1.lower(), fn2.lower()) >= 0.85:
                    cues.append(f"Nearly identical first names ({fn1}/{fn2})")

            # Family Name
            fam1, fam2 = str(row1.get("family_name", "")), str(row2.get("family_name", ""))
            if fam1 and fam2 and fam1.lower() == fam2.lower():
                cues.append("Same surname")

            # City
            c1, c2 = str(row1.get("city", "")), str(row2.get("city", ""))
            if c1 and c2 and c1 == c2:
                cues.append(f"Same city ({c1})")

            if cues:
                reasons_list.append("; ".join(cues[:3]))
            else:
                reasons_list.append("High overall demographic agreement across clinical attributes")
        else:
            reasons_list.append("Conflicting demographics or distinct patient profiles")

    # Assemble Scored Results DataFrame
    results_df = pd.DataFrame()
    results_df["record_id_1"] = r1_list
    results_df["record_id_2"] = r2_list
    results_df["duplicate_probability"] = [round(float(p), 4) for p in probs]
    results_df["primary_reason"] = reasons_list

    # Enrich with demographic context
    results_df["name_1"] = [f"{recs_dict.get(r, {}).get('first_name', '')} {recs_dict.get(r, {}).get('family_name', '')}".strip() for r in r1_list]
    results_df["name_2"] = [f"{recs_dict.get(r, {}).get('first_name', '')} {recs_dict.get(r, {}).get('family_name', '')}".strip() for r in r2_list]
    results_df["dob_1"] = [recs_dict.get(r, {}).get("date_of_birth", "") for r in r1_list]
    results_df["dob_2"] = [recs_dict.get(r, {}).get("date_of_birth", "") for r in r2_list]
    results_df["city_1"] = [recs_dict.get(r, {}).get("city", "") for r in r1_list]
    results_df["city_2"] = [recs_dict.get(r, {}).get("city", "") for r in r2_list]
    results_df["source_1"] = [recs_dict.get(r, {}).get("source_system", "") for r in r1_list]
    results_df["source_2"] = [recs_dict.get(r, {}).get("source_system", "") for r in r2_list]

    # Assign confidence labels based on Screen 5 thresholds
    def _assign_tier(p: float) -> str:
        if p >= active_thresh_strict:
            return "Very likely duplicate"
        elif p >= active_thresh:
            return "Worth reviewing"
        return "Unlikely duplicate"

    results_df["confidence_band"] = [_assign_tier(p) for p in probs]

    # Sort descending by duplicate probability
    results_df = results_df.sort_values(by="duplicate_probability", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1  # 1-indexed rank

    total_time = time.time() - t_start
    progress_bar.progress(100, text=f"Deduplication complete in {total_time:.2f}s!")
    status_msg.success(f"✅ Scored {n_candidates:,} candidate comparisons in **{total_time:.2f} seconds**.")

    # Save to session state
    st.session_state["batch_scored_results"] = results_df
    st.session_state["batch_scored_model"] = active_model
    st.session_state["batch_scored_file"] = source_label

# Retrieve results from session state
results_df = st.session_state["batch_scored_results"]
block_stats = st.session_state.get("batch_blocking_stats", {})

st.divider()

# ── Blocking Statistics for THIS File ─────────────────────────────────────────
st.subheader("3. Blocking Statistics for THIS Register")
st.caption("Quantifies the computational reduction achieved by candidate blocking prior to machine learning scoring:")

b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

with b_col1:
    st.metric("Total Records (N)", f"{block_stats.get('n_records', len(df_raw)):,}")

with b_col2:
    st.metric("All-Pairs Space", f"{block_stats.get('all_pairs_count', 0):,}", help="N(N-1)/2 unblocked search space.")

with b_col3:
    st.metric("Candidate Pairs", f"{block_stats.get('n_candidates', len(results_df)):,}", help="Union of 5 blocking passes.")

with b_col4:
    st.metric("Reduction Ratio", f"{block_stats.get('reduction_ratio', 0.0) * 100:.4f}%", help="1 - (Candidates / All-Pairs).")

with b_col5:
    n_flagged = int((results_df["duplicate_probability"] >= active_thresh).sum())
    st.metric(
        "Flagged Duplicates",
        f"{n_flagged:,}",
        delta=f"{(n_flagged / len(results_df) * 100):.2f}% of candidates",
        help="Pairs meeting or exceeding the current review threshold.",
    )

st.divider()

# ── Interactive Worklist Filters ──────────────────────────────────────────────
st.subheader("4. Prioritized Records Clerk Worklist")
st.caption(
    f"Candidate pairs ranked by duplicate probability under active policy (θ = {active_thresh:.2f}). "
    "Filter by confidence tier, facility, city, or minimum score."
)

f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    tier_filter = st.selectbox(
        "Filter by Confidence Tier",
        options=["All Bands", "Very likely duplicate", "Worth reviewing", "Unlikely duplicate"],
        index=0,
    )

with f_col2:
    all_sources = sorted(list(set(results_df["source_1"].dropna()) | set(results_df["source_2"].dropna())))
    selected_sources = st.multiselect("Filter by Source System", options=all_sources, default=[])

with f_col3:
    all_cities = sorted(list(set(results_df["city_1"].dropna()) | set(results_df["city_2"].dropna())))
    selected_cities = st.multiselect("Filter by City", options=all_cities, default=[])

with f_col4:
    min_score = st.slider("Min Probability", min_value=0.0, max_value=1.0, value=float(active_thresh), step=0.05)

# Check reviewed pairs in SQLite MergeStore
merge_store = MergeStore()
reviewed_pairs = merge_store.get_reviewed_pair_ids()

c_search, c_hide = st.columns([3, 1])
with c_search:
    search_query = st.text_input("🔎 Search Patient Name or Record ID", placeholder="e.g. Muhammad, REC001, Lahore").strip().lower()
with c_hide:
    hide_reviewed = st.checkbox(f"Hide decided ({len(reviewed_pairs)})", value=True, help="Hides candidate pairs that have already been merged, rejected, or deferred in Screen 6.")

# Apply Filters
filtered_df = results_df.copy()

if hide_reviewed and reviewed_pairs:
    is_reviewed = filtered_df.apply(
        lambda r: (min(str(r["record_id_1"]), str(r["record_id_2"])), max(str(r["record_id_1"]), str(r["record_id_2"]))) in reviewed_pairs,
        axis=1,
    )
    filtered_df = filtered_df[~is_reviewed]

if tier_filter != "All Bands":
    filtered_df = filtered_df[filtered_df["confidence_band"] == tier_filter]

if selected_sources:
    filtered_df = filtered_df[
        filtered_df["source_1"].isin(selected_sources) | filtered_df["source_2"].isin(selected_sources)
    ]

if selected_cities:
    filtered_df = filtered_df[
        filtered_df["city_1"].isin(selected_cities) | filtered_df["city_2"].isin(selected_cities)
    ]

if min_score > 0.0:
    filtered_df = filtered_df[filtered_df["duplicate_probability"] >= min_score]

if search_query:
    filtered_df = filtered_df[
        filtered_df["name_1"].str.lower().str.contains(search_query)
        | filtered_df["name_2"].str.lower().str.contains(search_query)
        | filtered_df["record_id_1"].str.lower().str.contains(search_query)
        | filtered_df["record_id_2"].str.lower().str.contains(search_query)
    ]

# Row Display Limit
disp_col1, disp_col2 = st.columns([1, 3])
with disp_col1:
    row_limit = st.selectbox("Display Limit", options=[25, 50, 100, 250, "All"], index=1)

display_count = len(filtered_df) if row_limit == "All" else int(row_limit)
view_df = filtered_df.iloc[:display_count]

st.markdown(f"Showing **{len(view_df):,}** of **{len(filtered_df):,}** matching candidate pairs (Total scored: {len(results_df):,}):")

# Format for Presentation
presentation_df = pd.DataFrame()
presentation_df["Rank"] = view_df.index
presentation_df["Tier"] = view_df["confidence_band"].map(
    {
        "Very likely duplicate": "🟢 High Confidence",
        "Worth reviewing": "🟡 Review Worklist",
        "Unlikely duplicate": "⚪ Distinct",
    }
)
presentation_df["Score"] = view_df["duplicate_probability"].apply(lambda p: f"{p*100:.1f}%")
presentation_df["Record 1"] = view_df["record_id_1"] + " | " + view_df["name_1"] + " (" + view_df["dob_1"] + ") [" + view_df["city_1"] + "]"
presentation_df["Record 2"] = view_df["record_id_2"] + " | " + view_df["name_2"] + " (" + view_df["dob_2"] + ") [" + view_df["city_2"] + "]"
presentation_df["Clerk Explanation"] = view_df["primary_reason"]

st.dataframe(presentation_df, use_container_width=True, hide_index=True)

st.write("")

# ── CSV Export ────────────────────────────────────────────────────────────────
export_col1, export_col2 = st.columns([1, 2])

with export_col1:
    # Full scored CSV download
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=True, index_label="rank")
    csv_data = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Full Scored Worklist (CSV)",
        data=csv_data,
        file_name=f"scored_duplicate_pairs_{active_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_col2:
    st.page_link("pages/6_Merge_Review.py", label="👉 Proceed to Screen 6: Golden Record Survivorship", icon="🔀")
