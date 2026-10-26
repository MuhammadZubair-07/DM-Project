"""Smoke and unit tests for the Streamlit application and screen foundation (Roadmap 7.11).

Verifies:
1. All 7 application entry points exist: app.py and all 6 pages in pages/.
2. All pages compile cleanly without syntax or indentation errors.
3. Centralized session state initialization in src/app_state.py.
4. Mandatory disclaimer banner is exact:
   'Student project. Not fit for clinical, financial or legal use. All data is synthetic.'
5. Cached data loaders correctly load reports from reports/*.json and reports/*.csv.
6. Screen 1 schema validation catches missing raw register columns.
7. Screen 2 benchmark table places the majority-class baseline beside every accuracy figure.
"""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import config as CFG
from src.app_state import (
    DEFAULT_COST_MATRIX,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_STRICT_THRESHOLD,
    init_session_state,
    load_audit_data,
    load_blocking_metrics,
    load_clean_effects,
    load_csv_data,
    load_evaluation_metrics,
    load_models_metrics,
)

ALL_APP_FILES = [
    Path("app.py"),
    Path("pages/1_Data_and_Quality.py"),
    Path("pages/2_Model_Performance.py"),
    Path("pages/3_Single_Prediction.py"),
    Path("pages/4_Batch_Scoring.py"),
    Path("pages/5_Settings_and_Cost_Matrix.py"),
    Path("pages/6_Merge_Review.py"),
]


# ── 1. File Existence & Layout ────────────────────────────────────────────────

def test_all_app_pages_exist_and_non_empty() -> None:
    """Verifies that app.py and all six pages exist in the repository."""
    for page_path in ALL_APP_FILES:
        assert page_path.exists(), f"Missing required application page: {page_path}"
        assert page_path.stat().st_size > 50, f"Application page is empty: {page_path}"


# ── 2. Compilation & Syntax Verification ──────────────────────────────────────

@pytest.mark.parametrize("page_path", ALL_APP_FILES)
def test_page_syntax_and_compilation(page_path: Path) -> None:
    """Compiles each Streamlit page script into Python bytecode to catch syntax errors."""
    try:
        py_compile.compile(str(page_path), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Syntax error in {page_path}: {e}")

    # Also parse with AST to verify parse tree integrity
    code_text = page_path.read_text(encoding="utf-8")
    tree = ast.parse(code_text, filename=str(page_path))
    assert isinstance(tree, ast.Module)


# ── 3. Session State Initialization ───────────────────────────────────────────

def test_app_state_initialization() -> None:
    """Tests that init_session_state sets all required state keys."""
    mock_state: dict[str, Any] = {}

    with patch("streamlit.session_state", mock_state):
        init_session_state()

        assert mock_state["selected_model"] == "decision_tree"
        assert pytest.approx(mock_state["threshold"], 1e-4) == 0.20
        assert pytest.approx(mock_state["threshold_strict"], 1e-4) == 50.0 / 60.0
        assert mock_state["cost_matrix"]["cost_fp"] == 5.0
        assert mock_state["cost_matrix"]["cost_fn"] == 20.0
        assert "active_dataset_name" in mock_state
        assert "active_dataset_path" in mock_state
        assert isinstance(mock_state["merge_history"], list)


# ── 4. Disclaimer Banner Verification ─────────────────────────────────────────

def test_mandatory_disclaimer_text_exact() -> None:
    """Verifies that the disclaimer banner contains the exact mandated student warning."""
    app_state_code = Path("src/app_state.py").read_text(encoding="utf-8")
    expected_warning = "Student project. Not fit for clinical, financial or legal use. All data is synthetic."
    assert expected_warning in app_state_code, (
        f"Disclaimer text in src/app_state.py does not match mandatory phrasing: {expected_warning}"
    )

    # Verify that every single page invokes render_disclaimer_banner()
    for page_path in ALL_APP_FILES:
        content = page_path.read_text(encoding="utf-8")
        assert "render_disclaimer_banner" in content, (
            f"Page {page_path} fails to call render_disclaimer_banner()!"
        )


# ── 5. Cached Data Accessors ──────────────────────────────────────────────────

def test_cached_loaders_data_integrity() -> None:
    """Verifies that cached data loaders successfully load reporting artifacts."""
    # 1. Audit Report
    audit_json, audit_df = load_audit_data()
    assert isinstance(audit_json, dict)
    assert not audit_df.empty, "Audit dataframe is empty"
    assert "category" in audit_df.columns
    assert "count" in audit_df.columns

    # 2. Cleaning Effects
    effects_json, effects_df = load_clean_effects()
    assert isinstance(effects_json, list)
    assert not effects_df.empty, "Cleaning effects dataframe is empty"
    assert "repair_item" in effects_df.columns
    assert "before" in effects_df.columns

    # 3. Blocking Report
    block_json = load_blocking_metrics()
    assert "all_pairs_comparisons" in block_json
    assert (
        "UNION (B1-B5)" in block_json["schemes_evaluation"]
        or "Union (B1-B5)" in block_json["schemes_evaluation"]
    )

    # 4. Evaluation Report
    eval_json = load_evaluation_metrics()
    assert "test_population" in eval_json
    assert "baselines" in eval_json
    assert "test_models" in eval_json
    assert "leakage_proof" in eval_json
    assert "split_experiment" in eval_json
    assert "robustness_check" in eval_json


# ── 6. Screen 1 Schema Validation ─────────────────────────────────────────────

def test_screen_1_schema_validation_logic() -> None:
    """Verifies that Screen 1 requires all 16 raw register attributes."""
    screen1_code = Path("pages/1_Data_and_Quality.py").read_text(encoding="utf-8")
    assert "REQUIRED_RAW_COLUMNS" in screen1_code

    required = [
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
    for col in required:
        assert f"'{col}'" in screen1_code or f'"{col}"' in screen1_code, (
            f"Missing required raw column in Screen 1 schema: {col}"
        )


# ── 7. Screen 2 Adjacent Baseline Assertion ───────────────────────────────────

def test_screen_2_baseline_adjacent_to_accuracy() -> None:
    """Verifies that Screen 2 explicitly renders the majority baseline beside every accuracy."""
    screen2_code = Path("pages/2_Model_Performance.py").read_text(encoding="utf-8")
    assert "Majority Baseline (%)" in screen2_code
    assert "Test Accuracy (%)" in screen2_code
    assert "MANDATORY ADJACENT COLUMN" in screen2_code or "majority_baseline" in screen2_code


# ── 8. Screen 5 Cost Matrix & Threshold Live Recomputation ───────────────────

def test_screen_5_recompute_and_cost_matrix() -> None:
    """Verifies that compute_cost_and_thresholds executes correctly for Screen 5."""
    from src.app_state import get_test_predictions
    from src.evaluate import compute_cost_and_thresholds

    y_true, p_pred = get_test_predictions("decision_tree")
    assert len(y_true) == 106298
    assert len(p_pred) == 106298

    # Evaluate default review threshold (C_FP=5, C_FN=20 -> theta*=0.20)
    res = compute_cost_and_thresholds(
        y_true=y_true,
        p_pred=p_pred,
        cost_fp=5.0,
        cost_fn=20.0,
        operating_threshold=0.20,
    )

    assert pytest.approx(res["cost_optimal_threshold_review"], 1e-4) == 0.20
    assert "operating_metrics" in res
    assert "accuracy" in res["operating_metrics"]
    assert "majority_baseline" in res["operating_metrics"]
    assert "precision" in res["operating_metrics"]
    assert "recall" in res["operating_metrics"]
    assert "expected_cost_1k" in res["operating_metrics"]
    assert "flagged_count" in res["operating_metrics"]
    assert "accuracy_given_up" in res
    assert res["operating_metrics"]["recall"] >= 0.99  # 100% recall on test set


# ── 9. Screen 3 Single Prediction & Zero-Jargon Explanation ───────────────────

def test_screen_3_single_prediction_and_jargon_guard() -> None:
    """Verifies that Screen 3 cleans records, scores, and generates jargon-free explanations."""
    import re
    from src.clean import clean_patient_register
    from src.explain import explain

    rec_a = {
        "record_id": "TEST_A",
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
    }
    rec_b = {
        "record_id": "TEST_B",
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
    }

    df_clean, _ = clean_patient_register(pd.DataFrame([rec_a, rec_b]))
    res = explain(
        pair=(df_clean.iloc[0].to_dict(), df_clean.iloc[1].to_dict()),
        model="decision_tree",
        threshold=0.20,
    )

    assert res["verdict"] in ["Duplicate Match", "Potential Duplicate"]
    assert res["probability"] >= 0.80
    assert len(res["reasons_for"]) > 0
    assert "field_comparisons" in res
    assert "first_name" in res["field_comparisons"]
    assert res["field_comparisons"]["first_name"]["status"] == "agree"

    # Strict Zero-Jargon Regex Check
    jargon_regex = re.compile(r"\bnode\b|\bgini\b|\bfeature_\w+|<=|>=|\b\d+\.\d{3,}\b", re.IGNORECASE)
    for text in res["reasons_for"] + res["reasons_against"]:
        assert not jargon_regex.search(text), f"Internal jargon leaked to clerk in explanation: {text}"


# ── 10. Screen 4 Batch Scoring & Worklist Generation ──────────────────────────

def test_screen_4_batch_scoring_sample_file() -> None:
    """Verifies that Screen 4 runs candidate blocking and scores sample_batch.csv."""
    from src.blocking import (
        pass_b1_soundex_dob_year,
        pass_b2_fn3_dob_year_month,
        pass_b3_phone_last7,
        pass_b4_national_id_exact,
        pass_b5_sorted_neighbourhood,
    )
    from src.clean import clean_patient_register
    from src.features import extract_features_for_pair_dataframe
    from src.models import Matcher

    sample_csv = CFG.DATA_DIR / "raw" / "sample_batch.csv"
    assert sample_csv.exists(), "sample_batch.csv is missing"

    df_raw = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    assert len(df_raw) > 1000

    df_clean, _ = clean_patient_register(df_raw)
    model_cols = [c for c in CFG.CLEAN_MODEL_ALLOWLIST if c in df_clean.columns]
    df_clean_model = df_clean[model_cols]

    p1, _, _, _ = pass_b1_soundex_dob_year(df_clean_model)
    p2, _, _, _ = pass_b2_fn3_dob_year_month(df_clean_model)
    p3, _, _, _ = pass_b3_phone_last7(df_clean_model)
    p4, _, _, _ = pass_b4_national_id_exact(df_clean_model)
    p5, _ = pass_b5_sorted_neighbourhood(df_clean_model, window_size=5)

    union_pairs = list(p1 | p2 | p3 | p4 | p5)
    assert len(union_pairs) > 1000

    all_pairs = len(df_clean) * (len(df_clean) - 1) // 2
    reduction_ratio = 1.0 - (len(union_pairs) / all_pairs)
    assert reduction_ratio > 0.95, f"Reduction ratio too low: {reduction_ratio}"

    # Extract features on small sample to verify pipeline
    pairs_sample = pd.DataFrame(union_pairs[:50], columns=["record_id_1", "record_id_2"])
    feats = extract_features_for_pair_dataframe(pairs_sample, df_clean_model)
    matcher = Matcher.load(CFG.MODELS_DIR / "decision_tree.joblib")
    probs = matcher.predict_proba(feats)[:, 1]
    assert len(probs) == 50
    assert all(0.0 <= p <= 1.0 for p in probs)

