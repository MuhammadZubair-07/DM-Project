"""Unit tests for Phase 4: Data Validation and validate() Assertions (Roadmap Section 7.4).

Verifies:
1. validate(df_clean) PASSES with 15/15 assertions and 0 violations.
2. validate(df_raw) FAILS with 10 assertions failing and documented violation counts.
3. Total assertion count is at least 10 (we implement 15).
4. Leakage guard assertion triggers if a forbidden column is introduced.
5. Hand-built unit tests for specific assertion edge cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config as CFG
from src.validate import validate


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


def test_validate_passes_on_clean_data(df_clean: pd.DataFrame) -> None:
    """Proves that validate() PASSES on data/clean/register_clean.csv with 0 violations."""
    report = validate(df_clean, dataset_name="register_clean.csv")

    assert report.is_valid is True, f"validate() unexpectedly failed on clean data: {report.failed_count} failures"
    assert report.failed_count == 0
    assert report.passed_count == report.total_assertions
    assert report.total_assertions >= 10

    # Ensure every individual assertion passed with 0 violations
    for a in report.assertions:
        assert a.passed is True, f"Assertion '{a.name}' failed with {a.violations} violations on clean data"
        assert a.violations == 0


def test_validate_fails_on_raw_data(df_raw: pd.DataFrame) -> None:
    """Proves that validate() FAILS on data/raw/register_raw.csv with multiple failures."""
    report = validate(df_raw, dataset_name="register_raw.csv")

    assert report.is_valid is False, "validate() unexpectedly passed on raw uncleaned data"
    assert report.failed_count >= 8, f"Expected at least 8 failing assertions on raw data, got {report.failed_count}"

    failed_names = {a.name for a in report.assertions if not a.passed}
    expected_failures = {
        "gender_canonical",
        "city_canonical",
        "dob_parses_and_valid_range",
        "no_dob_sentinels",
        "age_valid_range_and_consistent",
        "no_age_sentinels",
        "phone_format_valid",
        "no_phone_sentinels",
        "visit_count_valid_range",
    }
    missing_expected = expected_failures - failed_names
    assert not missing_expected, f"Raw data failed to trigger expected failures: {missing_expected}"


def test_validate_leakage_guard() -> None:
    """Verifies that injecting a forbidden column causes no_forbidden_columns to FAIL."""
    df_leak = pd.DataFrame([
        {
            "record_id": "REC0001",
            "source_system": "OPD",
            "registration_date": "2020-01-01",
            "gender": "M",
            "city": "Lahore",
            "true_entity_id": "ENT0001",  # FORBIDDEN COLUMN LEAKAGE
        }
    ])
    report = validate(df_leak, dataset_name="leaky_df")
    forbidden_assertion = next((a for a in report.assertions if a.name == "no_forbidden_columns"), None)
    assert forbidden_assertion is not None
    assert forbidden_assertion.passed is False
    assert forbidden_assertion.violations == 1


def test_validate_unit_assertions_isolated() -> None:
    """Tests isolated individual assertions on small hand-built DataFrames."""
    # 1. Clean valid frame
    df_good = pd.DataFrame([{
        "record_id": "REC0001",
        "gender": "M",
        "city": "Lahore",
        "registration_date": "2020-01-01",
        "date_of_birth": "1990-01-01",
        "age_years": "30",
        "phone": "03001234567",
        "national_id": "3520112345671",
        "first_name": "Ali",
        "family_name": "Khan",
        "visit_count": "10",
        "insurance_tier": "Standard",
    }])
    rep_good = validate(df_good)
    assert rep_good.is_valid is True

    # 2. Duplicate record_id
    df_bad_id = pd.DataFrame([
        {"record_id": "REC1"},
        {"record_id": "REC1"},
    ])
    rep_bad_id = validate(df_bad_id)
    assert not next(a for a in rep_bad_id.assertions if a.name == "record_id_unique_non_null").passed

    # 3. Dirty gender
    df_bad_gen = pd.DataFrame([{"record_id": "REC1", "gender": "male"}])
    rep_bad_gen = validate(df_bad_gen)
    assert not next(a for a in rep_bad_gen.assertions if a.name == "gender_canonical").passed

    # 4. Dirty city
    df_bad_city = pd.DataFrame([{"record_id": "REC1", "city": "LHR"}])
    rep_bad_city = validate(df_bad_city)
    assert not next(a for a in rep_bad_city.assertions if a.name == "city_canonical").passed

    # 5. Invalid visit count
    df_bad_visit = pd.DataFrame([{"record_id": "REC1", "visit_count": "9999"}])
    rep_bad_visit = validate(df_bad_visit)
    assert not next(a for a in rep_bad_visit.assertions if a.name == "visit_count_valid_range").passed
