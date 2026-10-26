"""Unit tests for Phase 2: Data profiling, measurement classification, and data dictionary.

Verifies:
1. reports/metrics.json exists and contains a well-structured 'profile' key.
2. reports/profile.md exists, is non-empty, and contains core markdown sections.
3. docs/data_dictionary.json exists, parses cleanly, and covers EVERY column in register_raw.csv.
4. All four Stevens measurement types (Nominal, Ordinal, Interval, Ratio) are present.
5. Ground truth / generator fault log fields are explicitly tagged as 'must-not-use'.
6. Identifiers and digit strings (national_id, phone) are classified as Nominal.
7. Profiling function can be executed without error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import config as CFG
from src.profile import COLUMN_SPECIFICATIONS, run_profiling
from src.utils import load_metrics


@pytest.fixture(scope="module")
def profile_metrics() -> dict[str, Any]:
    """Loads metrics.json and returns the 'profile' dictionary."""
    metrics = load_metrics()
    assert "profile" in metrics, "metrics.json missing required top-level key 'profile'"
    return metrics["profile"]


@pytest.fixture(scope="module")
def data_dictionary() -> dict[str, Any]:
    """Loads and returns docs/data_dictionary.json."""
    dict_path = CFG.DOCS_DIR / "data_dictionary.json"
    assert dict_path.exists(), f"Missing machine-readable dictionary at {dict_path}"
    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "data_dictionary.json must be a JSON object"
    return data


@pytest.fixture(scope="module")
def raw_columns() -> list[str]:
    """Reads header of register_raw.csv to get actual column list."""
    assert CFG.RAW_REGISTER_PATH.exists(), f"Missing raw register at {CFG.RAW_REGISTER_PATH}"
    df_head = pd.read_csv(CFG.RAW_REGISTER_PATH, nrows=1)
    return list(df_head.columns)


def test_metrics_profile_key(profile_metrics: dict[str, Any]) -> None:
    """Verifies reports/metrics.json has non-empty 'profile' key with expected fields."""
    assert isinstance(profile_metrics, dict)
    assert "shape" in profile_metrics
    assert "total_records" in profile_metrics
    assert "total_columns" in profile_metrics
    assert "measurement_types_present" in profile_metrics
    assert "columns" in profile_metrics
    assert profile_metrics["total_records"] > 0
    assert profile_metrics["total_columns"] == 16


def test_profile_md_exists_and_non_empty() -> None:
    """Verifies reports/profile.md exists, is non-empty, and contains key sections."""
    profile_md = CFG.REPORTS_DIR / "profile.md"
    assert profile_md.exists(), f"Missing profile report at {profile_md}"
    content = profile_md.read_text(encoding="utf-8")
    assert len(content.strip()) > 500, "reports/profile.md is too short or empty"
    assert "# Hospital Patient Register: Data Profile Report" in content
    assert "## 1. Executive Summary & Dataset Dimensions" in content
    assert "## 2. Column-by-Column Attribute Profile" in content
    assert "## 3. Lecture 2 Measurement Type Justifications" in content
    assert "## 4. Missingness Analysis" in content


def test_data_dictionary_covers_all_raw_columns(
    data_dictionary: dict[str, Any], raw_columns: list[str]
) -> None:
    """Confirms every column in register_raw.csv is fully documented in data_dictionary.json."""
    attributes = data_dictionary.get("attributes", {})
    assert len(attributes) >= len(raw_columns)

    missing_cols: list[str] = []
    for col in raw_columns:
        if col not in attributes:
            missing_cols.append(col)
        else:
            col_spec = attributes[col]
            assert "meaning" in col_spec and len(col_spec["meaning"]) > 0
            assert "measurement_type" in col_spec
            assert "scale_property" in col_spec
            assert "role" in col_spec
            assert "reasoning" in col_spec and len(col_spec["reasoning"]) > 0

    assert not missing_cols, f"Columns missing from data dictionary: {missing_cols}"


def test_all_four_stevens_measurement_types_present(data_dictionary: dict[str, Any]) -> None:
    """Verifies that Nominal, Ordinal, Interval, and Ratio scales are all represented."""
    attributes = data_dictionary.get("attributes", {})
    types_found = {spec["measurement_type"] for spec in attributes.values()}
    required_types = {"Nominal", "Ordinal", "Interval", "Ratio"}
    missing_types = required_types - types_found
    assert not missing_types, f"Data dictionary missing measurement types: {missing_types}"


def test_forbidden_columns_flagged_must_not_use(data_dictionary: dict[str, Any]) -> None:
    """Verifies true_entity_id and fault log objects are explicitly tagged as 'must-not-use'."""
    attributes = data_dictionary.get("attributes", {})
    forbidden_keys = ["true_entity_id", "fault_type", "original_value", "injected_value"]
    for key in forbidden_keys:
        assert key in attributes, f"Missing documentation for forbidden object: {key}"
        assert attributes[key]["role"] == "must-not-use", (
            f"Expected role 'must-not-use' for {key}, got '{attributes[key]['role']}'"
        )


def test_identifiers_and_digit_strings_classification(data_dictionary: dict[str, Any]) -> None:
    """Verifies that record_id, national_id, and phone are categorized as Nominal, not numeric."""
    attributes = data_dictionary.get("attributes", {})

    assert attributes["record_id"]["measurement_type"] == "Nominal"
    assert attributes["record_id"]["role"] == "identifier"

    assert attributes["national_id"]["measurement_type"] == "Nominal"
    assert attributes["national_id"]["role"] == "explanatory"

    assert attributes["phone"]["measurement_type"] == "Nominal"
    assert attributes["phone"]["role"] == "explanatory"


def test_ordinal_and_ratio_attributes_classification(data_dictionary: dict[str, Any]) -> None:
    """Verifies insurance_tier is Ordinal, and age_years and visit_count are Ratio."""
    attributes = data_dictionary.get("attributes", {})

    assert attributes["insurance_tier"]["measurement_type"] == "Ordinal"
    assert attributes["age_years"]["measurement_type"] == "Ratio"
    assert attributes["visit_count"]["measurement_type"] == "Ratio"
    assert attributes["registration_date"]["measurement_type"] == "Interval"
    assert attributes["date_of_birth"]["measurement_type"] == "Interval"


def test_data_dictionary_md_file_exists() -> None:
    """Verifies docs/data_dictionary.md exists and contains markdown documentation."""
    doc_path = CFG.DOCS_DIR / "data_dictionary.md"
    assert doc_path.exists(), f"Missing docs/data_dictionary.md at {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert "# Data Dictionary: Hospital Patient Register" in content
    assert "Stevens" in content
    assert "true_entity_id" in content


def test_leakage_statement_md_file_exists() -> None:
    """Verifies docs/leakage_statement.md exists and contains the required sections."""
    doc_path = CFG.DOCS_DIR / "leakage_statement.md"
    assert doc_path.exists(), f"Missing docs/leakage_statement.md at {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert "# Data Leakage Prevention Statement" in content
    assert "source_system" in content
    assert "true_entity_id" in content
    assert "Single-Table Register" in content or "unlinked flat table" in content


def test_run_profiling_programmatic_execution(tmp_path: Path) -> None:
    """Verifies run_profiling can execute cleanly and output to custom paths."""
    out_metrics = tmp_path / "temp_metrics.json"
    out_md = tmp_path / "temp_profile.md"
    out_dict = tmp_path / "temp_dict.json"

    res = run_profiling(
        metrics_path=out_metrics,
        report_md_path=out_md,
        dict_json_path=out_dict,
    )
    assert isinstance(res, dict)
    assert out_metrics.exists()
    assert out_md.exists()
    assert out_dict.exists()
