"""Unit tests for Phase 3: Data Quality Audit (Lecture 2).

Tests:
1. Every individual fault detector using tiny hand-built frames.
2. Verified generation of reports/audit.json, reports/audit.csv, and docs/quality_audit.md.
3. Verification that reports/metrics.json contains key 'audit'.
4. Verification that at least 8 Lecture 2 fault categories are present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import config as CFG
from src.audit import (
    detect_age_dob_inconsistency,
    detect_age_sentinel,
    detect_city_aliases,
    detect_collection_error_emergency_father,
    detect_collection_error_legacy_nid,
    detect_dirty_gender,
    detect_dob_after_registration,
    detect_dob_sentinel,
    detect_exact_duplicates,
    detect_impossible_age,
    detect_malformed_national_id,
    detect_name_case_whitespace,
    detect_name_typos,
    detect_non_iso_dates,
    detect_non_standard_phones,
    detect_phone_sentinel,
    detect_visit_outliers,
    detect_visit_sentinel,
    run_audit,
)
from src.utils import load_metrics


# ── Isolated Unit Tests on Tiny Hand-Built Frames ──────────────────────────────

def test_detect_exact_duplicates_unit() -> None:
    """Tests exact duplicate row detector on tiny 3-row frame."""
    df = pd.DataFrame([
        {"record_id": "REC1", "first_name": "Ali", "city": "Lahore"},
        {"record_id": "REC2", "first_name": "Ali", "city": "Lahore"},  # exact duplicate of row 0
        {"record_id": "REC3", "first_name": "Usman", "city": "Karachi"},
    ])
    mask = detect_exact_duplicates(df)
    assert not mask.iloc[0]
    assert mask.iloc[1]  # second copy marked as duplicate
    assert not mask.iloc[2]


def test_detect_city_aliases_unit() -> None:
    """Tests city alias detector with canonical and dirty city tokens."""
    s = pd.Series(["Lahore", "LHR", "lahore", "Karachi", "Khi", "Islamabad", ""])
    mask = detect_city_aliases(s)
    # Lahore, Karachi, Islamabad are canonical -> False
    # LHR, lahore, Khi are aliases -> True
    # '' is empty -> False
    assert not mask.iloc[0]  # Lahore
    assert mask.iloc[1]      # LHR
    assert mask.iloc[2]      # lahore
    assert not mask.iloc[3]  # Karachi
    assert mask.iloc[4]      # Khi
    assert not mask.iloc[5]  # Islamabad
    assert not mask.iloc[6]  # ''


def test_detect_dirty_gender_unit() -> None:
    """Tests dirty gender detector with M/F vs dirty variants."""
    s = pd.Series(["M", "F", "male", "female", "1", "2", "Ml", "Fm", ""])
    mask = detect_dirty_gender(s)
    assert not mask.iloc[0]  # M
    assert not mask.iloc[1]  # F
    assert mask.iloc[2]      # male
    assert mask.iloc[3]      # female
    assert mask.iloc[4]      # 1
    assert mask.iloc[5]      # 2
    assert mask.iloc[6]      # Ml
    assert mask.iloc[7]      # Fm
    assert not mask.iloc[8]  # ''


def test_detect_non_iso_dates_unit() -> None:
    """Tests date format detector for non-ISO representations."""
    s = pd.Series(["1995-04-12", "12/04/1995", "12-04-1995", "2020-01-01", ""])
    mask = detect_non_iso_dates(s)
    assert not mask.iloc[0]  # ISO
    assert mask.iloc[1]      # slash
    assert mask.iloc[2]      # dash
    assert not mask.iloc[3]  # ISO
    assert not mask.iloc[4]  # empty


def test_detect_non_standard_phones_unit() -> None:
    """Tests phone format detector with valid 11-digit vs formatted/sentinel."""
    s = pd.Series(["03001234567", "+923001234567", "0300-1234567", "0300 1234567", "00000000000", ""])
    mask = detect_non_standard_phones(s)
    assert not mask.iloc[0]  # standard clean
    assert mask.iloc[1]      # +92
    assert mask.iloc[2]      # dashes
    assert mask.iloc[3]      # spaces
    assert not mask.iloc[4]  # sentinel (handled separately)
    assert not mask.iloc[5]  # empty


def test_detect_name_case_whitespace_unit() -> None:
    """Tests untrimmed and irregular whitespace detector."""
    s = pd.Series(["Muhammad", " Ali", "Khan ", "Usman  Khan", "fatima", ""])
    mask = detect_name_case_whitespace(s)
    assert not mask.iloc[0]  # clean Title Case
    assert mask.iloc[1]      # leading space
    assert mask.iloc[2]      # trailing space
    assert mask.iloc[3]      # double internal space
    assert mask.iloc[4]      # lowercase
    assert not mask.iloc[5]  # empty


def test_detect_sentinels_unit() -> None:
    """Tests sentinel detectors for dates, ages, phones, and visits."""
    s_dob = pd.Series(["1990-05-15", "1900-01-01", "1970-01-01", ""])
    mask_dob = detect_dob_sentinel(s_dob)
    assert not mask_dob.iloc[0]
    assert mask_dob.iloc[1]
    assert mask_dob.iloc[2]

    s_age = pd.Series(["25", "-999", "-1", "999", "199", ""])
    mask_age = detect_age_sentinel(s_age)
    assert not mask_age.iloc[0]
    assert mask_age.iloc[1]
    assert mask_age.iloc[2]
    assert mask_age.iloc[3]
    assert mask_age.iloc[4]

    s_phone = pd.Series(["03001234567", "00000000000", "11111111111", ""])
    mask_phone = detect_phone_sentinel(s_phone)
    assert not mask_phone.iloc[0]
    assert mask_phone.iloc[1]
    assert mask_phone.iloc[2]

    s_visit = pd.Series(["5", "-999", "-1", "20", ""])
    mask_visit = detect_visit_sentinel(s_visit)
    assert not mask_visit.iloc[0]
    assert mask_visit.iloc[1]
    assert mask_visit.iloc[2]
    assert not mask_visit.iloc[3]


def test_detect_dob_after_registration_unit() -> None:
    """Tests detector for birth date occurring after registration date."""
    df = pd.DataFrame([
        {"date_of_birth": "1990-01-01", "registration_date": "2020-01-01"},  # valid
        {"date_of_birth": "2024-05-01", "registration_date": "2018-01-01"},  # impossible (DOB after reg)
        {"date_of_birth": "", "registration_date": "2020-01-01"},            # empty DOB
    ])
    mask = detect_dob_after_registration(df)
    assert not mask.iloc[0]
    assert mask.iloc[1]
    assert not mask.iloc[2]


def test_detect_impossible_age_unit() -> None:
    """Tests detector for age values outside [0, 110], excluding sentinels."""
    s = pd.Series(["25", "0", "110", "150", "-5", "-999", "-1", ""])
    mask = detect_impossible_age(s)
    assert not mask.iloc[0]  # 25 valid
    assert not mask.iloc[1]  # 0 valid
    assert not mask.iloc[2]  # 110 valid
    assert mask.iloc[3]      # 150 impossible
    assert mask.iloc[4]      # -5 impossible
    assert not mask.iloc[5]  # -999 is sentinel
    assert not mask.iloc[6]  # -1 is sentinel


def test_detect_age_dob_inconsistency_unit() -> None:
    """Tests cross-attribute consistency check between age_years and DOB."""
    df = pd.DataFrame([
        {"date_of_birth": "2000-01-01", "registration_date": "2020-01-01", "age_years": "20"},  # consistent
        {"date_of_birth": "2000-01-01", "registration_date": "2020-01-01", "age_years": "50"},  # inconsistent
        {"date_of_birth": "1900-01-01", "registration_date": "2020-01-01", "age_years": "25"},  # sentinel DOB
    ])
    mask = detect_age_dob_inconsistency(df)
    assert not mask.iloc[0]
    assert mask.iloc[1]
    assert not mask.iloc[2]


def test_detect_malformed_national_id_unit() -> None:
    """Tests national ID syntax validator (13 decimal digits)."""
    s = pd.Series(["3520112345671", "35201", "35201123456712", "35201ABC45671", "0000000000000", ""])
    mask = detect_malformed_national_id(s)
    assert not mask.iloc[0]  # clean 13 digits
    assert mask.iloc[1]      # 5 digits (too short)
    assert mask.iloc[2]      # 14 digits (too long)
    assert mask.iloc[3]      # contains letters
    assert not mask.iloc[4]  # sentinel
    assert not mask.iloc[5]  # empty


def test_detect_name_typos_unit() -> None:
    """Tests name typo detector against Pakistani base lexicon."""
    df = pd.DataFrame([
        {"first_name": "Muhammad", "family_name": "Khan"},     # clean in lexicon
        {"first_name": "Mohd", "family_name": "Khan"},         # abbreviation outside base lexicon
        {"first_name": "Zubair", "family_name": "Kahn"},       # typo in family name
        {"first_name": "Fatima", "family_name": "Ahmed"},      # clean in lexicon
    ])
    mask = detect_name_typos(df)
    assert not mask.iloc[0]
    assert mask.iloc[1]
    assert mask.iloc[2]
    assert not mask.iloc[3]


def test_detect_visit_outliers_unit() -> None:
    """Tests robust visit outlier detector."""
    # A distribution centered around 25 with an extreme 9999 outlier
    base_visits = [20, 22, 24, 25, 26, 28, 30] * 10
    s = pd.Series(base_visits + [9999, -999])
    mask = detect_visit_outliers(s.astype(str))
    assert not mask.iloc[0]
    assert mask.iloc[-2]     # 9999 is outlier
    assert not mask.iloc[-1] # -999 is sentinel


def test_detect_collection_errors_unit() -> None:
    """Tests systemic collection error detectors."""
    df = pd.DataFrame([
        {"source_system": "Legacy", "national_id": "", "father_name": "Ali Khan"},
        {"source_system": "OPD", "national_id": "3520112345671", "father_name": "Ali Khan"},
        {"source_system": "Emergency", "national_id": "3520112345671", "father_name": ""},
    ])
    mask_legacy = detect_collection_error_legacy_nid(df)
    mask_emergency = detect_collection_error_emergency_father(df)

    assert mask_legacy.iloc[0]
    assert not mask_legacy.iloc[1]
    assert not mask_legacy.iloc[2]

    assert not mask_emergency.iloc[0]
    assert not mask_emergency.iloc[1]
    assert mask_emergency.iloc[2]


# ── Full Audit Artifact Verification Tests ─────────────────────────────────────

def test_audit_json_file_exists_and_valid() -> None:
    """Verifies reports/audit.json exists and contains structured audit records."""
    audit_json_path = CFG.REPORTS_DIR / "audit.json"
    assert audit_json_path.exists(), f"Missing audit JSON at {audit_json_path}"
    with open(audit_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "total_records" in data
    assert "total_fault_types_detected" in data
    assert "faults" in data
    assert "missingness_dependency_analysis" in data
    assert "detector_evaluation_vs_fault_log" in data
    assert data["total_fault_types_detected"] >= 8


def test_audit_csv_file_exists_and_valid() -> None:
    """Verifies reports/audit.csv exists and has required schema."""
    audit_csv_path = CFG.REPORTS_DIR / "audit.csv"
    assert audit_csv_path.exists(), f"Missing audit CSV at {audit_csv_path}"
    df = pd.read_csv(audit_csv_path)

    expected_cols = {
        "fault_code", "category", "fault_name", "affected_columns",
        "detecting_expression", "count", "percentage", "proposed_action", "reason"
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 8


def test_quality_audit_md_report_exists_and_complete() -> None:
    """Verifies docs/quality_audit.md exists, is non-empty, and has required sections."""
    audit_md_path = CFG.DOCS_DIR / "quality_audit.md"
    assert audit_md_path.exists(), f"Missing quality audit markdown at {audit_md_path}"
    content = audit_md_path.read_text(encoding="utf-8")

    assert len(content.strip()) > 1000
    assert "# Data Quality Audit Report" in content
    assert "## 1. Executive Summary & Audit Methodology" in content
    assert "## 2. Fault Catalogue Table" in content
    assert "## 4. Missingness Analysis & Source-System Dependency Proof" in content
    assert "## 5. Detector Performance vs. Ground-Truth Fault Log" in content
    assert "## 6. Near-Duplicates are the Target, Not a Repair" in content


def test_audit_key_in_metrics_json() -> None:
    """Verifies reports/metrics.json contains 'audit' key with summary metrics."""
    metrics = load_metrics()
    assert "audit" in metrics, "Missing 'audit' key in reports/metrics.json"
    audit_metric = metrics["audit"]
    assert "faults_detected_count" in audit_metric
    assert "total_affected_counts" in audit_metric
    assert "chi2_missingness_significant_columns" in audit_metric
    assert len(audit_metric["chi2_missingness_significant_columns"]) > 0


def test_run_audit_custom_execution(tmp_path: Path) -> None:
    """Verifies run_audit can be called programmatically with custom output paths."""
    out_json = tmp_path / "test_audit.json"
    out_csv = tmp_path / "test_audit.csv"
    out_md = tmp_path / "test_audit.md"

    res = run_audit(
        json_path=out_json,
        csv_out_path=out_csv,
        md_out_path=out_md,
    )
    assert isinstance(res, dict)
    assert out_json.exists()
    assert out_csv.exists()
    assert out_md.exists()
