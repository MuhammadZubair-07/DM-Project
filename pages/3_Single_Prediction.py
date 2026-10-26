"""Screen 3: Single Pair Duplicate Prediction & Clinical Explanation.

Enables hospital records clerks to:
1. Input or inspect any two patient charts (Record A and Record B).
2. One-click load pre-curated test scenarios (Easy match, Hard negative sibling pair,
   Subtle phonetic duplicate, Emergency missing-data admission, Distinct patients).
3. Standardize and clean inputs through the exact same cleaning pipeline as training.
4. Score with the active classifier chosen in Screen 5.
5. Display duplicate probability, cost-derived confidence tier, verdict badge,
   structured plain-English reasons FOR and AGAINST, and per-field agreement flags.
"""

from __future__ import annotations

import re
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

st.set_page_config(
    page_title="Single Prediction - Patient Deduplication",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_disclaimer_banner()
render_help_sidebar()

# ── Title & Introduction ───────────────────────────────────────────────────────
st.title("🔍 Screen 3: Single Pair Duplicate Prediction & Clinical Explanation")
st.markdown(
    "Compare two patient demographic charts side-by-side. The system runs the production "
    "cleaning and feature extraction pipeline, applies the active classifier from Screen 5, "
    "and produces **jargon-free clinical explanations** tailored for medical records clerks."
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
    f"([Modify in Screen 5](pages/5_Settings_and_Cost_Matrix.py))"
)
st.divider()

# ── Pre-Curated Clinical Test Cases ───────────────────────────────────────────
EXAMPLE_CASES = {
    "easy_match": {
        "title": "🟢 Easy Duplicate (Casing, Typo & Phone Reformat)",
        "desc": "Same patient with minor name casing, space irregularity, and dashed phone format.",
        "rec_a": {
            "record_id": "REC-A101",
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
            "record_id": "REC-B101",
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
    "hard_negative_siblings": {
        "title": "🔴 Hard Negative (Twin Sisters / Same Father & Address)",
        "desc": "Twin sisters sharing father name, residential address, contact number, and birth date, but distinct individuals.",
        "rec_a": {
            "record_id": "REC-A202",
            "first_name": "Zainab",
            "family_name": "Tariq",
            "father_name": "Tariq Mahmood",
            "gender": "Female",
            "date_of_birth": "2002-08-14",
            "age_years": "22",
            "national_id": "35202-7654321-2",
            "phone": "0321-9876543",
            "address": "House 14-B Gulberg III",
            "city": "Lahore",
            "source_system": "Pediatric Ward",
            "registration_date": "2023-11-10",
            "blood_group": "A+",
            "insurance_tier": "Silver",
            "visit_count": "2",
        },
        "rec_b": {
            "record_id": "REC-B202",
            "first_name": "Fatima",
            "family_name": "Tariq",
            "father_name": "Tariq Mahmood",
            "gender": "F",
            "date_of_birth": "2002-08-14",
            "age_years": "22",
            "national_id": "35202-7654321-4",
            "phone": "0321-9876543",
            "address": "House 14-B Gulberg III",
            "city": "Lahore",
            "source_system": "General Medicine",
            "registration_date": "2024-02-18",
            "blood_group": "O+",
            "insurance_tier": "Silver",
            "visit_count": "1",
        },
    },
    "subtle_duplicate": {
        "title": "🟡 Subtle Duplicate (Transposed Surname & Phone Digit)",
        "desc": "Same patient where first/family names were inverted during triage intake.",
        "rec_a": {
            "record_id": "REC-A303",
            "first_name": "Usman",
            "family_name": "Ali",
            "father_name": "Ahmad Raza",
            "gender": "Male",
            "date_of_birth": "1990-11-23",
            "age_years": "33",
            "national_id": "42101-9988776-3",
            "phone": "0333-4567890",
            "address": "Flat 4, Clifton Block 2",
            "city": "Karachi",
            "source_system": "Emergency Room",
            "registration_date": "2024-03-01",
            "blood_group": "O-",
            "insurance_tier": "Bronze",
            "visit_count": "6",
        },
        "rec_b": {
            "record_id": "REC-B303",
            "first_name": "Ali",
            "family_name": "Usman",
            "father_name": "Ahmad Raza",
            "gender": "M",
            "date_of_birth": "1990-11-23",
            "age_years": "33",
            "national_id": "42101-9988776-3",
            "phone": "0333-4567809",
            "address": "Flat 4, Clifton Block 2",
            "city": "Karachi",
            "source_system": "Cardiology Lab",
            "registration_date": "2024-07-12",
            "blood_group": "O-",
            "insurance_tier": "Bronze",
            "visit_count": "2",
        },
    },
    "missing_emergency": {
        "title": "⚠️ Emergency Intake (Unrecorded CNIC & Missing Phone)",
        "desc": "Trauma patient admitted unconscious; national ID and phone unrecorded at bedside.",
        "rec_a": {
            "record_id": "REC-A404",
            "first_name": "Bilawal",
            "family_name": "Raza",
            "father_name": "Raza Hussain",
            "gender": "Male",
            "date_of_birth": "1978-04-03",
            "age_years": "46",
            "national_id": "",
            "phone": "",
            "address": "Sector G-9/2",
            "city": "Islamabad",
            "source_system": "Trauma Emergency",
            "registration_date": "2024-05-10",
            "blood_group": "AB+",
            "insurance_tier": "Standard",
            "visit_count": "1",
        },
        "rec_b": {
            "record_id": "REC-B404",
            "first_name": "Bilawal",
            "family_name": "Raza",
            "father_name": "Raza Hussain",
            "gender": "M",
            "date_of_birth": "1978-04-03",
            "age_years": "46",
            "national_id": "37405-5544332-1",
            "phone": "0345-1122334",
            "address": "Sector G-9/2 Street 14",
            "city": "Islamabad",
            "source_system": "Internal Medicine",
            "registration_date": "2023-09-22",
            "blood_group": "AB+",
            "insurance_tier": "Standard",
            "visit_count": "5",
        },
    },
    "distinct_patients": {
        "title": "⚪ Clear Distinct Patients (Different People)",
        "desc": "Completely unrelated patients registered in different cities with different demographics.",
        "rec_a": {
            "record_id": "REC-A505",
            "first_name": "Ayesha",
            "family_name": "Siddiqa",
            "father_name": "Muhammad Siddiq",
            "gender": "Female",
            "date_of_birth": "1992-03-15",
            "age_years": "32",
            "national_id": "35201-1111111-2",
            "phone": "0301-1111111",
            "address": "Jaranwala Road",
            "city": "Faisalabad",
            "source_system": "Maternity Ward",
            "registration_date": "2024-02-14",
            "blood_group": "B-",
            "insurance_tier": "Gold",
            "visit_count": "3",
        },
        "rec_b": {
            "record_id": "REC-B505",
            "first_name": "Imran",
            "family_name": "Farooq",
            "father_name": "Farooq Ahmad",
            "gender": "Male",
            "date_of_birth": "1980-10-20",
            "age_years": "44",
            "national_id": "35202-2222222-1",
            "phone": "0302-2222222",
            "address": "Bosan Road",
            "city": "Multan",
            "source_system": "Orthopedic Center",
            "registration_date": "2024-04-18",
            "blood_group": "A-",
            "insurance_tier": "Standard",
            "visit_count": "1",
        },
    },
}

st.subheader("1. Quick-Load Clinical Scenarios")
st.caption("Click any test case to populate both patient charts automatically:")

q_cols = st.columns(5)
for idx, (case_key, case_data) in enumerate(EXAMPLE_CASES.items()):
    with q_cols[idx]:
        if st.button(case_data["title"], key=f"btn_{case_key}", use_container_width=True):
            st.session_state["single_rec_a"] = dict(case_data["rec_a"])
            st.session_state["single_rec_b"] = dict(case_data["rec_b"])
            st.session_state["current_case_note"] = case_data["desc"]
            st.rerun()

if "current_case_note" in st.session_state:
    st.info(f"💡 **Loaded Scenario:** {st.session_state['current_case_note']}")

# Initialize default records if empty
if "single_rec_a" not in st.session_state:
    st.session_state["single_rec_a"] = dict(EXAMPLE_CASES["easy_match"]["rec_a"])
if "single_rec_b" not in st.session_state:
    st.session_state["single_rec_b"] = dict(EXAMPLE_CASES["easy_match"]["rec_b"])

rec_a = st.session_state["single_rec_a"]
rec_b = st.session_state["single_rec_b"]

st.divider()

# ── Side-by-Side Demographic Form ─────────────────────────────────────────────
st.subheader("2. Patient Demographic Charts")
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("### 📋 Record A (Chart 1)")
    r_id_a = st.text_input("Record ID (A)", value=rec_a.get("record_id", "REC001"), key="f_id_a")
    fn_a = st.text_input("First Name (A)", value=rec_a.get("first_name", ""), placeholder="e.g. Muhammad", key="f_fn_a")
    fam_a = st.text_input("Family Name / Surname (A)", value=rec_a.get("family_name", ""), placeholder="e.g. Tariq", key="f_fam_a")
    fath_a = st.text_input("Father Name (A)", value=rec_a.get("father_name", ""), placeholder="e.g. Tariq Khan", key="f_fath_a")
    
    sub_a1, sub_a2 = st.columns(2)
    with sub_a1:
        gen_a = st.text_input("Gender (A)", value=rec_a.get("gender", "M"), placeholder="M / F / Male / Female", key="f_gen_a")
    with sub_a2:
        dob_a = st.text_input("Date of Birth (A)", value=rec_a.get("date_of_birth", "1985-05-12"), placeholder="YYYY-MM-DD", key="f_dob_a")

    sub_a3, sub_a4 = st.columns(2)
    with sub_a3:
        age_a = st.text_input("Age in Years (A)", value=str(rec_a.get("age_years", "39")), key="f_age_a")
    with sub_a4:
        bg_a = st.text_input("Blood Group (A)", value=rec_a.get("blood_group", "B+"), placeholder="e.g. B+, O-", key="f_bg_a")

    nid_a = st.text_input("National ID / CNIC (A)", value=rec_a.get("national_id", ""), placeholder="35201-1234567-1", key="f_nid_a")
    ph_a = st.text_input("Phone Number (A)", value=rec_a.get("phone", ""), placeholder="0300-1234567", key="f_ph_a")
    city_a = st.text_input("City (A)", value=rec_a.get("city", "Lahore"), placeholder="e.g. Lahore, Karachi", key="f_city_a")
    addr_a = st.text_input("Residential Address (A)", value=rec_a.get("address", ""), placeholder="Street, Sector, City", key="f_addr_a")

    sub_a5, sub_a6 = st.columns(2)
    with sub_a5:
        sys_a = st.text_input("Source System (A)", value=rec_a.get("source_system", "Hospital Central"), key="f_sys_a")
    with sub_a6:
        vis_a = st.text_input("Visit Count (A)", value=str(rec_a.get("visit_count", "4")), key="f_vis_a")

with col_b:
    st.markdown("### 📋 Record B (Chart 2)")
    r_id_b = st.text_input("Record ID (B)", value=rec_b.get("record_id", "REC002"), key="f_id_b")
    fn_b = st.text_input("First Name (B)", value=rec_b.get("first_name", ""), placeholder="e.g. Mohammad", key="f_fn_b")
    fam_b = st.text_input("Family Name / Surname (B)", value=rec_b.get("family_name", ""), placeholder="e.g. Tariq", key="f_fam_b")
    fath_b = st.text_input("Father Name (B)", value=rec_b.get("father_name", ""), placeholder="e.g. Tariq Khan", key="f_fath_b")

    sub_b1, sub_b2 = st.columns(2)
    with sub_b1:
        gen_b = st.text_input("Gender (B)", value=rec_b.get("gender", "M"), placeholder="M / F / Male / Female", key="f_gen_b")
    with sub_b2:
        dob_b = st.text_input("Date of Birth (B)", value=rec_b.get("date_of_birth", "1985-05-12"), placeholder="YYYY-MM-DD", key="f_dob_b")

    sub_b3, sub_b4 = st.columns(2)
    with sub_b3:
        age_b = st.text_input("Age in Years (B)", value=str(rec_b.get("age_years", "39")), key="f_age_b")
    with sub_b4:
        bg_b = st.text_input("Blood Group (B)", value=rec_b.get("blood_group", "B+"), placeholder="e.g. B+, O-", key="f_bg_b")

    nid_b = st.text_input("National ID / CNIC (B)", value=rec_b.get("national_id", ""), placeholder="35201-1234567-1", key="f_nid_b")
    ph_b = st.text_input("Phone Number (B)", value=rec_b.get("phone", ""), placeholder="0300-1234567", key="f_ph_b")
    city_b = st.text_input("City (B)", value=rec_b.get("city", "Lahore"), placeholder="e.g. Lahore, Karachi", key="f_city_b")
    addr_b = st.text_input("Residential Address (B)", value=rec_b.get("address", ""), placeholder="Street, Sector, City", key="f_addr_b")

    sub_b5, sub_b6 = st.columns(2)
    with sub_b5:
        sys_b = st.text_input("Source System (B)", value=rec_b.get("source_system", "Outpatient Clinic"), key="f_sys_b")
    with sub_b6:
        vis_b = st.text_input("Visit Count (B)", value=str(rec_b.get("visit_count", "1")), key="f_vis_b")

st.divider()

# ── Execution Trigger ──────────────────────────────────────────────────────────
exec_col1, exec_col2 = st.columns([1, 3])
with exec_col1:
    run_prediction = st.button("🚀 Compare & Explain Pair", type="primary", use_container_width=True)

# Build current dicts from input widgets
current_a = {
    "record_id": r_id_a or "REC_A",
    "first_name": fn_a,
    "family_name": fam_a,
    "father_name": fath_a,
    "gender": gen_a,
    "date_of_birth": dob_a,
    "age_years": age_a,
    "national_id": nid_a,
    "phone": ph_a,
    "city": city_a,
    "address": addr_a,
    "blood_group": bg_a,
    "insurance_tier": rec_a.get("insurance_tier", "Standard"),
    "visit_count": vis_a,
    "source_system": sys_a or "System A",
    "registration_date": rec_a.get("registration_date", "2024-01-01"),
}

current_b = {
    "record_id": r_id_b or "REC_B",
    "first_name": fn_b,
    "family_name": fam_b,
    "father_name": fath_b,
    "gender": gen_b,
    "date_of_birth": dob_b,
    "age_years": age_b,
    "national_id": nid_b,
    "phone": ph_b,
    "city": city_b,
    "address": addr_b,
    "blood_group": bg_b,
    "insurance_tier": rec_b.get("insurance_tier", "Standard"),
    "visit_count": vis_b,
    "source_system": sys_b or "System B",
    "registration_date": rec_b.get("registration_date", "2024-01-01"),
}

# Update state
st.session_state["single_rec_a"] = current_a
st.session_state["single_rec_b"] = current_b

# ── Pipeline Execution & Results Display ──────────────────────────────────────
# Run prediction automatically if pre-loaded or button clicked
if run_prediction or "single_scored" in st.session_state or True:
    st.subheader("3. Clinical Prediction & Jargon-Free Explanation")

    # Step A: Clean records through production cleaner
    df_raw_pair = pd.DataFrame([current_a, current_b])
    try:
        df_clean_pair, _ = clean_patient_register(df_raw_pair)
        clean_rec_a = df_clean_pair.iloc[0].to_dict()
        clean_rec_b = df_clean_pair.iloc[1].to_dict()
    except Exception as e:
        st.error(f"Error during data cleaning pipeline: {e}")
        st.stop()

    # Step B: Score and Explain
    try:
        matcher = get_matcher(active_model)
        explanation = explain(
            pair=(clean_rec_a, clean_rec_b),
            model=matcher,
            threshold=active_thresh,
            threshold_strict=active_thresh_strict,
            record_1=clean_rec_a,
            record_2=clean_rec_b,
        )
    except Exception as e:
        st.error(f"Error running model scoring and explanation engine: {e}")
        st.stop()

    prob = explanation["probability"]
    conf_label = explanation["confidence_label"]
    verdict = explanation["verdict"]

    # 1. Verdict Banner
    if verdict == "Duplicate Match":
        st.success(
            f"### 🟢 Verdict: Duplicate Match (High Confidence)\n\n"
            f"**Recommendation:** Recommended for Golden Record Merge. Probability (**{prob * 100:.1f}%**) "
            f"exceeds the strict merge boundary ($\\theta_{{strict}} = {active_thresh_strict:.2f}$).",
            icon="✅",
        )
    elif verdict == "Potential Duplicate":
        st.warning(
            f"### 🟡 Verdict: Potential Duplicate (Needs Review)\n\n"
            f"**Recommendation:** Dispatched to Records Clerk Worklist. Probability (**{prob * 100:.1f}%**) "
            f"exceeds review threshold ($\\theta = {active_thresh:.2f}$) but requires manual clerk confirmation.",
            icon="⚠️",
        )
    else:
        st.info(
            f"### ⚪ Verdict: Distinct Patients (Unlikely Duplicate)\n\n"
            f"**Recommendation:** Keep Patient Records Separate. Probability (**{prob * 100:.1f}%**) "
            f"is below the review threshold ($\\theta = {active_thresh:.2f}$).",
            icon="ℹ️",
        )

    # 2. Key Metrics Bar
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Duplicate Probability", f"{prob * 100:.2f}%", help="Calibrated likelihood that both records refer to the same individual.")
        st.progress(float(prob))
    with res_c2:
        st.metric("Confidence Tier", conf_label, help="Bayesian cost-derived confidence band.")
    with res_c3:
        st.metric("Operating Threshold", f"θ = {active_thresh:.2f}", help="Review cutoff configured in Screen 5.")
    with res_c4:
        st.metric("Scoring Model", model_display, help="Active classifier family.")

    st.write("")

    # 3. Plain-English Reasons (Zero Jargon)
    reasons_col1, reasons_col2 = st.columns(2, gap="medium")

    with reasons_col1:
        with st.container(border=True):
            st.markdown("#### ✅ Reasons Supporting Match (Reasons FOR)")
            reasons_for = explanation.get("reasons_for", [])
            if reasons_for:
                for r in reasons_for:
                    st.markdown(f"- {r}")
            else:
                st.caption("No specific matching attributes strongly supported duplicate status.")

    with reasons_col2:
        with st.container(border=True):
            st.markdown("#### ⚠️ Reasons Opposing Match (Reasons AGAINST)")
            reasons_against = explanation.get("reasons_against", [])
            if reasons_against:
                for r in reasons_against:
                    st.markdown(f"- {r}")
            else:
                st.caption("No conflicting demographic discrepancies detected.")

    st.caption(f"📝 **Synthesis:** {explanation.get('clerk_summary', '')}")
    st.divider()

    # 4. Field-by-Field Agreement Table (Screen 6 Preview)
    st.subheader("4. Field-by-Field Alignment (Cleaned Demographics)")
    st.caption("Review side-by-side attributes after standardized cleaning and formatting:")

    field_comps = explanation.get("field_comparisons", {})
    table_rows = []

    for field_key, comp in field_comps.items():
        status = comp.get("status", "unknown")
        if status == "agree":
            badge = "🟢 Agree"
        elif status == "conflict":
            badge = "🔴 Conflict"
        else:
            badge = "🟡 Unknown / Missing"

        table_rows.append(
            {
                "Demographic Attribute": comp.get("field_label", field_key),
                "Record A (Cleaned)": str(comp.get("value_1", "")),
                "Record B (Cleaned)": str(comp.get("value_2", "")),
                "Alignment Status": badge,
                "Clinical Observation": comp.get("detail", ""),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Link to merge screen
    st.markdown("---")
    c_link1, c_link2 = st.columns(2)
    with c_link1:
        st.page_link("pages/4_Batch_Scoring.py", label="👉 Proceed to Screen 4: Batch Register Scoring", icon="📋")
    with c_link2:
        st.page_link("pages/6_Merge_Review.py", label="👉 Proceed to Screen 6: Golden Record Survivorship", icon="🔀")
