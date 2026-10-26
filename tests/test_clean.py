"""Unit tests for Phase 4: Data Cleaning, Provenance, and Effects (Lecture 2).

Tests:
1. data/clean/register_clean.csv exists, has correct row count, and has clean columns.
2. Original raw values are preserved in *_raw columns.
3. Provenance columns exist for every repaired field and have no null/empty entries.
4. Anomaly flags exist and correctly flag sentinels and outliers.
5. reports/clean_effects.json and reports/clean_effects.csv exist with before/after counts.
6. reports/metrics.json has key 'cleaning'.
7. Near-duplicates are strictly preserved (not deduplicated in cleaning).
8. docs/cleaning_report.md exists and is non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import config as CFG
from src.utils import load_metrics


@pytest.fixture(scope="module")
def df_clean() -> pd.DataFrame:
    """Loads clean register DataFrame."""
    assert CFG.CLEAN_REGISTER_PATH.exists(), f"Missing clean register at {CFG.CLEAN_REGISTER_PATH}"
    return pd.read_csv(CFG.CLEAN_REGISTER_PATH, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def df_raw() -> pd.DataFrame:
    """Loads raw register DataFrame."""
    assert CFG.RAW_REGISTER_PATH.exists(), f"Missing raw register at {CFG.RAW_REGISTER_PATH}"
    return pd.read_csv(CFG.RAW_REGISTER_PATH, dtype=str, keep_default_na=False)


def test_clean_file_dimensions(df_clean: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Verifies clean register has identical row count and includes all required columns."""
    assert len(df_clean) == len(df_raw), "Clean register row count must match raw register"
    assert len(df_clean) >= 50000

    # Model allowlist columns must all be present
    for col in CFG.CLEAN_MODEL_ALLOWLIST:
        assert col in df_clean.columns, f"Clean register missing model column: {col}"


def test_raw_columns_preserved(df_clean: pd.DataFrame) -> None:
    """Verifies that original uncleaned values are preserved in *_raw columns."""
    raw_cols = [
        "first_name_raw", "family_name_raw", "father_name_raw", "gender_raw",
        "city_raw", "date_of_birth_raw", "age_years_raw", "phone_raw",
        "national_id_raw", "address_raw", "blood_group_raw", "insurance_tier_raw",
        "visit_count_raw",
    ]
    for col in raw_cols:
        assert col in df_clean.columns, f"Missing preserved raw column: {col}"


def test_provenance_columns_valid(df_clean: pd.DataFrame) -> None:
    """Verifies every provenance column exists and has non-empty values for all rows."""
    for prov_col in CFG.CLEAN_PROVENANCE_COLUMNS:
        assert prov_col in df_clean.columns, f"Missing provenance column: {prov_col}"
        empty_count = (df_clean[prov_col].str.strip() == "").sum()
        assert empty_count == 0, f"Provenance column {prov_col} contains {empty_count} blank rows"


def test_anomaly_indicator_flags(df_clean: pd.DataFrame) -> None:
    """Verifies boolean indicator flags exist and capture sentinels and outliers."""
    for flag_col in CFG.CLEAN_FLAG_COLUMNS:
        assert flag_col in df_clean.columns, f"Missing flag column: {flag_col}"
        # Values must be boolean or string booleans
        unique_vals = set(df_clean[flag_col].astype(str).unique())
        assert unique_vals.issubset({"True", "False"}), f"Invalid flag values in {flag_col}: {unique_vals}"

    assert (df_clean["dob_sentinel_flag"].astype(str) == "True").sum() > 0
    assert (df_clean["phone_sentinel_flag"].astype(str) == "True").sum() > 0
    assert (df_clean["visit_outlier_flag"].astype(str) == "True").sum() > 0


def test_clean_effects_files_exist_and_populated() -> None:
    """Verifies reports/clean_effects.json and .csv exist and contain before/after counts."""
    assert CFG.CLEAN_EFFECTS_PATH.exists(), f"Missing effects JSON at {CFG.CLEAN_EFFECTS_PATH}"
    assert CFG.CLEAN_EFFECTS_CSV_PATH.exists(), f"Missing effects CSV at {CFG.CLEAN_EFFECTS_CSV_PATH}"

    with open(CFG.CLEAN_EFFECTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 10

    for item in data:
        assert "repair_item" in item
        assert "before" in item
        assert "after" in item
        assert "change" in item
        assert isinstance(item["before"], int)
        assert isinstance(item["after"], int)


def test_metrics_json_has_cleaning_key() -> None:
    """Verifies reports/metrics.json contains 'cleaning' key with summary stats."""
    metrics = load_metrics()
    assert "cleaning" in metrics, "Missing 'cleaning' key in metrics.json"
    clean_metric = metrics["cleaning"]
    assert "total_records_cleaned" in clean_metric
    assert clean_metric["total_records_cleaned"] > 0
    assert "repairs_summary" in clean_metric


def test_near_duplicates_strictly_preserved(df_clean: pd.DataFrame) -> None:
    """Asserts that duplicate rows were not unilaterally dropped or merged during cleaning."""
    gt_path = CFG.GROUND_TRUTH_PATH
    if gt_path.exists():
        df_gt = pd.read_csv(gt_path)
        # Count entities with > 1 record in ground truth
        vc = df_gt["true_entity_id"].value_counts()
        dupe_entities = set(vc[vc > 1].index)
        dupe_rids = set(df_gt[df_gt["true_entity_id"].isin(dupe_entities)]["record_id"])

        # All duplicate record_ids must still be present in df_clean
        clean_rids = set(df_clean["record_id"])
        missing_dupes = dupe_rids - clean_rids
        assert len(missing_dupes) == 0, f"Cleaning mistakenly dropped {len(missing_dupes)} candidate duplicate records"


def test_cleaning_report_markdown_exists() -> None:
    """Verifies docs/cleaning_report.md exists and contains before/after table."""
    report_path = CFG.DOCS_DIR / "cleaning_report.md"
    assert report_path.exists(), f"Missing docs/cleaning_report.md at {report_path}"
    content = report_path.read_text(encoding="utf-8")
    assert "# Data Cleaning and Provenance Report" in content
    assert "## 2. Before and After Cleaning Comparison Table" in content
    assert "## 3. Provenance Audit Distributions" in content
