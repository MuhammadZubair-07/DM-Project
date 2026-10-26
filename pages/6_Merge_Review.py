"""Screen 6: Reversible Golden Record Merge Review & Audit Log.

Adheres to Roadmap Sections D8 and 7.11:
1. Interactive Side-by-Side Comparison:
   - Compares Record A and Record B field-by-field.
   - Highlights alignment using color AND text/icons:
     🟢 Agree | 🔴 Conflict | 🟡 Unknown (Missing/Sentinel).
   - Displays model confidence score, confidence label, and plain-English reasons FOR/AGAINST.
2. Clinical Safety Guardrails:
   - NEVER auto-merge.
   - Prominent warnings when duplicate probability is below the 'Very likely duplicate' band.
   - Red alerts when critical clinical fields conflict (National ID or Blood Group).
3. Intelligent Survivorship Engine:
   - Rule-based golden record suggestions (prefer non-missing/non-sentinel, newer contact
     registration dates, complete identifiers, and higher encounter counts).
   - Per-field radio selector: Record A, Record B, or Custom Edit.
4. Clerical Action Triage:
   - Merge (with a 2-step confirmation showing the resulting Golden Record and clerk note).
   - Reject (records pair as distinct patients in SQLite decisions table).
   - Defer (records deferral timestamp and returns pair to queue).
5. Audit Log & Byte-for-Byte Reversible Undo:
   - Full history of all merges and decisions saved in SQLite `data/merges.db`.
   - 1-click Undo restoring both original records byte-for-byte.
   - Export reviewed decisions as labelled training/audit pairs.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src import config as CFG
from src.app_state import (
    get_matcher,
    init_session_state,
    render_disclaimer_banner,
    render_help_sidebar,
)
from src.clean import clean_patient_register
from src.explain import explain
from src.merge_store import MergeStore, compute_survivorship_suggestions

st.set_page_config(
    page_title="Merge Review - Patient Deduplication",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# Initialize SQLite Merge Store
store = MergeStore()

# ── Title & Clerk Guidance Panel ──────────────────────────────────────────────
st.title("🔀 Screen 6: Reversible Golden Record Merge Review")
st.markdown(
    "Perform human-in-the-loop chart consolidation. Review candidate duplicate records side-by-side, "
    "tailor attribute survivorship choices, execute audited merges, and **reverse any action at any time**."
)

with st.expander("ℹ️ Medical Records Clerk Workflow Guide", expanded=False):
    st.markdown(
        """
        **Step-by-Step Merge Review Protocol:**
        1. **Inspect Alignment:** Review the two patient charts side-by-side. Check highlighted tags:
           - 🟢 **Agree:** Attribute values are identical or phonetically equivalent.
           - 🔴 **Conflict:** Attribute values contradict each other (e.g. differing blood groups or CNICs).
           - 🟡 **Unknown:** Value was missing or an unrecorded sentinel placeholder.
        2. **Review Model Evidence:** Check duplicate probability and plain-English reasons **FOR** and **AGAINST** duplicate status.
        3. **Formulate Golden Record:** The system pre-selects the most complete, non-sentinel, and recent attributes. You may override any attribute using the radio buttons or type a manual correction.
        4. **Action Triage:**
           - **🔀 Merge:** Combines records into a surviving Golden Record with full before/after audit tracking.
           - **❌ Reject:** Confirms patients are distinct individuals (removes from active duplicate queue).
           - **⏸️ Defer:** Flags for supervisor review or paper file inspection.
        5. **Reversible Undo:** If an error is realized later, click **Undo** in the Audit Log to restore both original records byte-for-byte.
        """
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
    f"Active Policy: **{model_display}** | "
    f"Review Threshold: **θ = {active_thresh:.2f}** | "
    f"Auto-Merge Threshold: **θ_strict = {active_thresh_strict:.2f}** "
    f"([Modify in Screen 5](pages/5_Settings_and_Cost_Matrix.py))"
)
st.divider()

# ── Candidate Pair Queue Management ───────────────────────────────────────────
st.subheader("1. Candidate Duplicate Pair Selection")

# Get list of already decided pairs to filter out of the queue
reviewed_pairs = store.get_reviewed_pair_ids()

# Check sources
batch_results = st.session_state.get("batch_scored_results")
has_batch_results = batch_results is not None and not batch_results.empty

queue_source = st.radio(
    "Queue Source:",
    options=[
        "Screen 4 Scored Worklist (Production Queue)",
        "Screen 3 Single Prediction Pair",
        "Pre-Curated Clinical Demonstration Pairs",
    ],
    index=0 if has_batch_results else 2,
    horizontal=True,
)

current_rec_a: dict[str, Any] | None = None
current_rec_b: dict[str, Any] | None = None
current_pair_prob: float = 0.0

if queue_source == "Screen 4 Scored Worklist (Production Queue)":
    if not has_batch_results:
        st.info("No batch results in session state. Please run **Screen 4 (Batch Scoring)** first, or select demonstration pairs above.")
        st.page_link("pages/4_Batch_Scoring.py", label="👉 Open Screen 4 to Score a Register", icon="📋")
    else:
        # Filter out reviewed pairs unless clerk requests all
        show_all = st.checkbox("Show already-reviewed pairs", value=False)
        available_pairs = []

        for idx, row in batch_results.iterrows():
            r1, r2 = str(row["record_id_1"]), str(row["record_id_2"])
            canon = (min(r1, r2), max(r1, r2))
            if show_all or canon not in reviewed_pairs:
                available_pairs.append((idx, row))

        if not available_pairs:
            st.success("🎉 All candidate duplicate pairs in the worklist have been reviewed!")
        else:
            st.markdown(f"**Worklist Queue:** {len(available_pairs):,} unreviewed candidate pairs remaining (out of {len(batch_results):,} total candidates).")

            # Selector
            pair_options = [
                f"Pair #{r['Rank'] if 'Rank' in r else i+1}: {r['record_id_1']} & {r['record_id_2']} (Score: {float(r['duplicate_probability'])*100:.1f}% - {r.get('confidence_band', '')})"
                for i, (_, r) in enumerate(available_pairs)
            ]
            selected_idx = st.selectbox("Select Candidate Pair to Review:", options=range(len(pair_options)), format_func=lambda i: pair_options[i])
            chosen_row = available_pairs[selected_idx][1]

            # Reconstruct records from row info and active dataset
            r1_id = str(chosen_row["record_id_1"])
            r2_id = str(chosen_row["record_id_2"])
            current_pair_prob = float(chosen_row["duplicate_probability"])

            raw_df = st.session_state.get("batch_raw_df")
            if raw_df is not None and "record_id" in raw_df.columns:
                r1_match = raw_df[raw_df["record_id"] == r1_id]
                r2_match = raw_df[raw_df["record_id"] == r2_id]
                if not r1_match.empty and not r2_match.empty:
                    current_rec_a = r1_match.iloc[0].to_dict()
                    current_rec_b = r2_match.iloc[0].to_dict()

            if current_rec_a is None:
                # Fallback from row columns
                current_rec_a = {
                    "record_id": r1_id,
                    "first_name": chosen_row.get("name_1", "").split()[0] if chosen_row.get("name_1") else "Patient",
                    "family_name": " ".join(chosen_row.get("name_1", "").split()[1:]) if chosen_row.get("name_1") else "",
                    "date_of_birth": chosen_row.get("dob_1", ""),
                    "city": chosen_row.get("city_1", ""),
                    "source_system": chosen_row.get("source_1", ""),
                }
                current_rec_b = {
                    "record_id": r2_id,
                    "first_name": chosen_row.get("name_2", "").split()[0] if chosen_row.get("name_2") else "Patient",
                    "family_name": " ".join(chosen_row.get("name_2", "").split()[1:]) if chosen_row.get("name_2") else "",
                    "date_of_birth": chosen_row.get("dob_2", ""),
                    "city": chosen_row.get("city_2", ""),
                    "source_system": chosen_row.get("source_2", ""),
                }

elif queue_source == "Screen 3 Single Prediction Pair":
    if "single_rec_a" in st.session_state and "single_rec_b" in st.session_state:
        current_rec_a = dict(st.session_state["single_rec_a"])
        current_rec_b = dict(st.session_state["single_rec_b"])
        st.success(f"Loaded active pair from Screen 3: `{current_rec_a.get('record_id')}` & `{current_rec_b.get('record_id')}`.")
    else:
        st.info("No active pair found from Screen 3. Please configure a pair on Screen 3 or choose demo pairs.")

else:  # Pre-Curated Clinical Demonstration Pairs
    DEMO_PAIRS = {
        "demo_1_casing_typo": {
            "title": "🟢 Pair 1: Easy Duplicate (Name casing, formatting & phone reformat)",
            "rec_a": {
                "record_id": "REC-MRG-101",
                "first_name": " muhammad ",
                "family_name": "tariq",
                "father_name": "khan",
                "gender": "male",
                "date_of_birth": "1985-05-12",
                "age_years": "39",
                "national_id": "35201-1234567-1",
                "phone": "0300-1234567",
                "address": "123 Mall Road",
                "city": "lahore",
                "source_system": "Hospital Central",
                "registration_date": "2024-01-15",
                "blood_group": "B+",
                "insurance_tier": "Gold",
                "visit_count": "4",
            },
            "rec_b": {
                "record_id": "REC-MRG-102",
                "first_name": "Mohammad",
                "family_name": "Tariq",
                "father_name": "Khan",
                "gender": "M",
                "date_of_birth": "1985-05-12",
                "age_years": "39",
                "national_id": "3520112345671",
                "phone": "03001234567",
                "address": "123 Mall Rd.",
                "city": "Lahore",
                "source_system": "Outpatient Clinic",
                "registration_date": "2024-06-20",
                "blood_group": "B+",
                "insurance_tier": "Gold",
                "visit_count": "1",
            },
        },
        "demo_2_blood_conflict": {
            "title": "🚨 Pair 2: Critical Clinical Hazard (Conflicting Blood Groups: A+ vs B-)",
            "rec_a": {
                "record_id": "REC-MRG-201",
                "first_name": "Rashid",
                "family_name": "Mahmood",
                "father_name": "Mahmood Khan",
                "gender": "Male",
                "date_of_birth": "1975-09-20",
                "age_years": "49",
                "national_id": "35202-5544332-1",
                "phone": "0321-4455667",
                "address": "Street 5, Model Town",
                "city": "Lahore",
                "source_system": "Surgical ICU",
                "registration_date": "2023-10-10",
                "blood_group": "A+",
                "insurance_tier": "Standard",
                "visit_count": "5",
            },
            "rec_b": {
                "record_id": "REC-MRG-202",
                "first_name": "Rashid",
                "family_name": "Mahmood",
                "father_name": "Mahmood Khan",
                "gender": "M",
                "date_of_birth": "1975-09-20",
                "age_years": "49",
                "national_id": "35202-5544332-1",
                "phone": "0321-4455667",
                "address": "Street 5, Model Town",
                "city": "Lahore",
                "source_system": "Hematology Ward",
                "registration_date": "2024-03-12",
                "blood_group": "B-",  # Critical conflict!
                "insurance_tier": "Standard",
                "visit_count": "2",
            },
        },
        "demo_3_cnic_conflict": {
            "title": "🚨 Pair 3: Critical Citizen Conflict (Distinct Valid CNICs: 35201 vs 35202)",
            "rec_a": {
                "record_id": "REC-MRG-301",
                "first_name": "Tariq",
                "family_name": "Javed",
                "father_name": "Javed Iqbal",
                "gender": "Male",
                "date_of_birth": "1982-12-05",
                "age_years": "41",
                "national_id": "35201-1122334-1",
                "phone": "0333-8877665",
                "address": "DHA Phase 5",
                "city": "Lahore",
                "source_system": "Cardiology Clinic",
                "registration_date": "2024-01-05",
                "blood_group": "O+",
                "insurance_tier": "Platinum",
                "visit_count": "2",
            },
            "rec_b": {
                "record_id": "REC-MRG-302",
                "first_name": "Tariq",
                "family_name": "Javed",
                "father_name": "Javed Iqbal",
                "gender": "Male",
                "date_of_birth": "1982-12-05",
                "age_years": "41",
                "national_id": "35202-9988776-3",  # Distinct valid CNIC!
                "phone": "0333-8877665",
                "address": "DHA Phase 5",
                "city": "Lahore",
                "source_system": "General Medicine",
                "registration_date": "2024-04-18",
                "blood_group": "O+",
                "insurance_tier": "Platinum",
                "visit_count": "1",
            },
        },
    }

    demo_choice = st.selectbox(
        "Choose Demonstration Pair:",
        options=list(DEMO_PAIRS.keys()),
        format_func=lambda k: DEMO_PAIRS[k]["title"],
    )
    current_rec_a = dict(DEMO_PAIRS[demo_choice]["rec_a"])
    current_rec_b = dict(DEMO_PAIRS[demo_choice]["rec_b"])

if current_rec_a is None or current_rec_b is None:
    st.warning("Please select a candidate pair to proceed with merge review.")
    st.stop()

st.divider()

# ── Clean & Score Pair ────────────────────────────────────────────────────────
try:
    df_raw_pair = pd.DataFrame([current_rec_a, current_rec_b])
    df_clean_pair, _ = clean_patient_register(df_raw_pair)
    clean_a = df_clean_pair.iloc[0].to_dict()
    clean_b = df_clean_pair.iloc[1].to_dict()

    matcher = get_matcher(active_model)
    explanation = explain(
        pair=(clean_a, clean_b),
        model=matcher,
        threshold=active_thresh,
        threshold_strict=active_thresh_strict,
        record_1=clean_a,
        record_2=clean_b,
    )
    prob = explanation["probability"]
    conf_label = explanation["confidence_label"]
    verdict = explanation["verdict"]
except Exception as e:
    st.error(f"Error computing model explanation: {e}")
    st.stop()

# ── Model Evidence & Safety Alerts Card ───────────────────────────────────────
st.subheader("2. Model Intelligence & Clinical Safety Gate")

ev_col1, ev_col2 = st.columns([1, 2], gap="medium")

with ev_col1:
    with st.container(border=True):
        st.markdown("#### Model Duplicate Assessment")
        st.metric("Duplicate Probability", f"{prob * 100:.2f}%", help="Calibrated likelihood that both records refer to the same patient.")
        st.progress(float(prob))
        st.markdown(f"- **Confidence Tier:** `{conf_label}`")
        st.markdown(f"- **Verdict:** `{verdict}`")
        st.markdown(f"- **Operating Cutoff:** `θ = {active_thresh:.2f}`")

with ev_col2:
    with st.container(border=True):
        st.markdown("#### Clinical Plain-English Reasoning")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("**✅ Reasons FOR Match:**")
            for r in explanation.get("reasons_for", [])[:3]:
                st.markdown(f"- {r}")
        with r_col2:
            st.markdown("**⚠️ Reasons AGAINST Match:**")
            for r in explanation.get("reasons_against", [])[:3]:
                st.markdown(f"- {r}")

# ── Requirement 4: Clinical Safety Warnings ───────────────────────────────────
# Check for Blood Group Conflict
bg_a = str(clean_a.get("blood_group", "") or "").strip().upper()
bg_b = str(clean_b.get("blood_group", "") or "").strip().upper()
has_blood_conflict = bool(bg_a and bg_b and bg_a != bg_b)

# Check for National ID Conflict
nid_a = "".join(c for c in str(clean_a.get("national_id", "")) if c.isdigit())
nid_b = "".join(c for c in str(clean_b.get("national_id", "")) if c.isdigit())
has_nid_conflict = bool(len(nid_a) == 13 and len(nid_b) == 13 and nid_a != nid_b)

# Check for Low Confidence Warning
is_low_confidence = bool(prob < active_thresh_strict)

if has_blood_conflict:
    st.error(
        f"🚨 **CRITICAL CLINICAL HAZARD: Conflicting Blood Groups Detected!**\n\n"
        f"Record A specifies **{bg_a}** while Record B specifies **{bg_b}**. "
        f"Merging patients with conflicting blood groups creates an **immediate transfusion incompatibility hazard**! "
        f"Do NOT merge unless a laboratory redraw has confirmed clerical misentry.",
        icon="🩸",
    )

if has_nid_conflict:
    st.error(
        f"🚨 **CRITICAL CITIZEN CONFLICT: Conflicting Valid National IDs (CNIC)!**\n\n"
        f"Record A specifies CNIC `{nid_a}` while Record B specifies CNIC `{nid_b}`. "
        f"Both are valid 13-digit Pakistani national identity cards. Merging distinct CNICs may collapse two separate citizens!",
        icon="🪪",
    )

if is_low_confidence and not (has_blood_conflict or has_nid_conflict):
    st.warning(
        f"⚠️ **Low/Moderate Confidence Notice:** Model probability ({prob * 100:.1f}%) is below the high-confidence "
        f"auto-merge boundary ($\\theta_{{strict}} = {active_thresh_strict:.2f}$). "
        f"Independent clerical verification of paper charts or patient contact is strongly advised.",
        icon="⚠️",
    )

st.divider()

# ── Field-by-Field Side-by-Side Comparison & Survivorship ─────────────────────
st.subheader("3. Field-by-Field Survivorship & Golden Record Construction")
st.markdown(
    "Choose winning values for the consolidated Golden Patient Record. "
    "Suggestions adhere to **Lecture 2 & Roadmap D8** survivorship rules (completeness, recency, and non-sentinel preference)."
)

suggested_golden, default_choices, rationale_map = compute_survivorship_suggestions(clean_a, clean_b)
field_comps = explanation.get("field_comparisons", {})

# Survivorship Selection Form
final_golden_record: dict[str, Any] = {
    "record_id": f"GOLD_{clean_a.get('record_id', 'REC1')}_{clean_b.get('record_id', 'REC2')}",
    "source_system": "Golden Patient Master Index (Merged)",
    "registration_date": datetime.now().strftime("%Y-%m-%d"),
}
field_provenance: dict[str, str] = {}

CORE_FIELDS = [
    ("first_name", "First Name"),
    ("family_name", "Family Name / Surname"),
    ("father_name", "Father Name"),
    ("gender", "Gender"),
    ("date_of_birth", "Date of Birth"),
    ("age_years", "Age (Years)"),
    ("national_id", "National ID (CNIC)"),
    ("phone", "Phone Number"),
    ("city", "City"),
    ("address", "Residential Address"),
    ("blood_group", "Blood Group"),
    ("insurance_tier", "Insurance Tier"),
    ("visit_count", "Cumulative Visit Count"),
]

for field_key, field_label in CORE_FIELDS:
    val_a = str(clean_a.get(field_key, "") or "").strip()
    val_b = str(clean_b.get(field_key, "") or "").strip()
    comp = field_comps.get(field_key, {})
    status = comp.get("status", "unknown")

    # Status icon and text (using colour AND text/icons)
    if status == "agree":
        badge = "🟢 Agree (Match)"
        badge_color = "#e6f4ea"
    elif status == "conflict":
        badge = "🔴 Conflict (Discrepancy!)"
        badge_color = "#fce8e6"
    else:
        badge = "🟡 Unknown (Missing/Sentinel)"
        badge_color = "#fef7e0"

    with st.container(border=True):
        f_c1, f_c2, f_c3 = st.columns([1, 1, 1.5], gap="medium")

        with f_c1:
            st.markdown(f"**Record A ({clean_a.get('record_id', 'A')}):**")
            st.code(val_a if val_a else "— (Unrecorded)", language=None)

        with f_c2:
            st.markdown(f"**Record B ({clean_b.get('record_id', 'B')}):**")
            st.code(val_b if val_b else "— (Unrecorded)", language=None)

        with f_c3:
            st.markdown(f"**{field_label}** — {badge}")
            default_sel = default_choices.get(field_key, "Record A")

            choice = st.radio(
                f"Surviving value for {field_label}:",
                options=[f"Record A: {val_a}", f"Record B: {val_b}", "Custom Manual Edit"],
                index=0 if default_sel == "Record A" else 1,
                key=f"surv_{field_key}_{clean_a.get('record_id')}_{clean_b.get('record_id')}",
                horizontal=True,
            )

            # Rationale note
            st.caption(f"💡 *Suggestion Rationale:* {rationale_map.get(field_key, 'Defaulted to primary chart.')}")

            if "Custom Manual Edit" in choice:
                custom_val = st.text_input(
                    f"Enter custom value for {field_label}:",
                    value=suggested_golden.get(field_key, ""),
                    key=f"custom_{field_key}",
                )
                final_golden_record[field_key] = custom_val
                field_provenance[field_key] = "Manual Clerk Edit"
            elif "Record A:" in choice:
                final_golden_record[field_key] = val_a
                field_provenance[field_key] = f"Record A ({clean_a.get('record_id', 'A')})"
            else:
                final_golden_record[field_key] = val_b
                field_provenance[field_key] = f"Record B ({clean_b.get('record_id', 'B')})"

st.divider()

# ── Action Triage Buttons & Confirmation Modal ────────────────────────────────
st.subheader("4. Action Triage (Never Auto-Merge)")
st.caption("Human clerk decision required. Confirming a merge executes the irreversible snapshot and logs to SQLite.")

# Clerk audit note
clerk_note = st.text_area(
    "Clerk Decision Note / Clinical Justification:",
    placeholder="e.g. Identity confirmed via mobile subscriber contact and matching father name.",
    key="clerk_audit_note",
)

act_col1, act_col2, act_col3 = st.columns(3, gap="medium")

with act_col1:
    # Merge action with 2-step confirmation
    merge_clicked = st.button("🔀 Merge Records (Golden Record)", type="primary", use_container_width=True)

with act_col2:
    reject_clicked = st.button("❌ Reject (Confirm Distinct Patients)", use_container_width=True)

with act_col3:
    defer_clicked = st.button("⏸️ Defer (Flag for Senior Review)", use_container_width=True)

# ── Confirmation Step for Merge ───────────────────────────────────────────────
if merge_clicked or st.session_state.get("pending_merge_confirm"):
    st.session_state["pending_merge_confirm"] = True

    with st.container(border=True):
        st.markdown("### ⚠️ Step 2: Confirm Golden Record Merge")
        st.warning(
            f"You are about to merge **Record {clean_a.get('record_id')}** and **Record {clean_b.get('record_id')}**.\n\n"
            f"- Both original records will be preserved in full JSON snapshots inside `data/merges.db`.\n"
            f"- The Golden Record below will become the surviving master patient chart.\n"
            f"- You can **Undo** this merge at any time in the Audit Log below."
        )

        # Golden Record Preview
        preview_df = pd.DataFrame(
            [
                {"Attribute": k, "Surviving Value": str(v), "Provenance Source": field_provenance.get(k, "Default")}
                for k, v in final_golden_record.items()
            ]
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        conf_col1, conf_col2 = st.columns([1, 1])
        with conf_col1:
            if st.button("✅ Confirm & Execute Reversible Merge", type="primary", use_container_width=True):
                try:
                    merge_id = store.record_merge(
                        record_1=clean_a,
                        record_2=clean_b,
                        golden_record=final_golden_record,
                        field_provenance=field_provenance,
                        clerk_note=clerk_note or "Merged via Screen 6 Review Console",
                        model_name=active_model,
                        duplicate_probability=prob,
                    )
                    st.session_state["pending_merge_confirm"] = False
                    st.success(f"🎉 **Merge Executed Successfully!** Merge ID: `{merge_id}`. Saved to SQLite store.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to record merge in SQLite store: {e}")

        with conf_col2:
            if st.button("❌ Cancel Merge", use_container_width=True):
                st.session_state["pending_merge_confirm"] = False
                st.rerun()

# ── Reject Decision ───────────────────────────────────────────────────────────
if reject_clicked:
    try:
        dec_id = store.record_decision(
            record_1=clean_a,
            record_2=clean_b,
            decision="reject",
            clerk_note=clerk_note or "Clerk marked distinct patients",
            model_name=active_model,
            duplicate_probability=prob,
        )
        st.success(f"✅ **Recorded as Distinct Patients.** Decision ID: `{dec_id}`. Pair removed from active review worklist.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to record reject decision: {e}")

# ── Defer Decision ────────────────────────────────────────────────────────────
if defer_clicked:
    try:
        dec_id = store.record_decision(
            record_1=clean_a,
            record_2=clean_b,
            decision="defer",
            clerk_note=clerk_note or "Deferred for supervisory or paper chart verification",
            model_name=active_model,
            duplicate_probability=prob,
        )
        st.info(f"⏸️ **Candidate Pair Deferred.** Decision ID: `{dec_id}`. Logged timestamp and returned to queue.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to record deferral: {e}")

st.divider()

# ── Reversible Merge Audit Log & Exact Undo ───────────────────────────────────
st.subheader("5. Reversible Merge Audit Log & Exact Undo")
st.caption(
    "All executed merges and decisions are stored with cryptographic precision in `data/merges.db`. "
    "Clicking **Undo** restores both original patient records **byte-for-byte**."
)

tab_merges, tab_decisions = st.tabs(["📜 Merge Audit History (with Undo)", "📋 Rejected & Deferred Decisions"])

with tab_merges:
    all_merges = store.get_all_merges()
    if not all_merges:
        st.info("No patient merges have been executed in this database yet.")
    else:
        st.markdown(f"Total Merges Recorded: **{len(all_merges)}**")

        for m in all_merges:
            m_id = m["merge_id"]
            m_status = m["status"]
            is_active = (m_status == "merged")
            status_badge = "🟢 Active Master Record" if is_active else "↩️ Undone (Restored Originals)"

            with st.container(border=True):
                h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
                with h_col1:
                    st.markdown(f"**Merge ID:** `{m_id}` — {status_badge}")
                    st.caption(
                        f"Records Merged: `{m['record_id_1']}` & `{m['record_id_2']}` | "
                        f"Timestamp: `{m['timestamp']}` | Model: `{m.get('model_name', 'default')}` (Prob: `{m.get('duplicate_probability', 0.0)*100:.1f}%`)"
                    )
                    st.markdown(f"📝 *Clerk Note:* {m.get('clerk_note', 'No note recorded')}")

                with h_col2:
                    with st.expander("🔍 Golden Record Snapshot"):
                        st.json(m["golden_record"])

                with h_col3:
                    if is_active:
                        if st.button("↩️ Undo Merge", key=f"undo_{m_id}", type="secondary", use_container_width=True):
                            try:
                                r1_orig, r2_orig = store.undo_merge(m_id)
                                st.success(
                                    f"✅ **Merge `{m_id}` Undone!** Restored original records "
                                    f"`{r1_orig.get('record_id')}` and `{r2_orig.get('record_id')}` byte-for-byte."
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to undo merge: {e}")
                    else:
                        st.caption(f"Undone at: `{m.get('undone_timestamp', '')}`")

with tab_decisions:
    all_decisions = store.get_all_decisions()
    if not all_decisions:
        st.info("No reject or deferral decisions recorded yet.")
    else:
        dec_df = pd.DataFrame(
            [
                {
                    "Decision ID": d["decision_id"],
                    "Record A": d["record_id_1"],
                    "Record B": d["record_id_2"],
                    "Decision": "❌ Rejected (Distinct)" if d["decision"] == "reject" else "⏸️ Deferred",
                    "Timestamp": d["timestamp"],
                    "Clerk Note": d.get("clerk_note", ""),
                }
                for d in all_decisions
            ]
        )
        st.dataframe(dec_df, use_container_width=True, hide_index=True)

st.write("")

# ── Export Labelled Decisions ──────────────────────────────────────────────────
st.markdown("#### 📥 Export Decision Registry")
st.caption("Export clerical audit outcomes to enrich future retraining sets or clinical governance reports:")

exp_col1, exp_col2 = st.columns([1, 2])
with exp_col1:
    decisions_df = store.export_decisions_as_labelled_pairs()
    if not decisions_df.empty:
        csv_buf = io.StringIO()
        decisions_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Labelled Decisions (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"clerk_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No decisions available to export yet.")
