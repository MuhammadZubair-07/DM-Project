"""Tests for src/generate_data.py.

Tests validate:
  1. The three output files exist and have expected shape.
  2. ground_truth.csv is NOT in register_raw.csv.
  3. record_id is unique, shuffled (no correlation with entity).
  4. Hard-negative counts are in the sanity report.
  5. The trivial same-DOB+phone rule does NOT achieve near-perfect recall.
  6. All four measurement types are present in the register.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# ── Use the project's config for canonical paths ──────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as CFG


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def register() -> pd.DataFrame:
    """Load the raw register (generated once per test session)."""
    if not CFG.RAW_REGISTER_PATH.exists():
        pytest.skip("register_raw.csv not found; run python -m src.generate_data first")
    return pd.read_csv(CFG.RAW_REGISTER_PATH, dtype=str, low_memory=False)


@pytest.fixture(scope="module")
def ground_truth() -> pd.DataFrame:
    if not CFG.GROUND_TRUTH_PATH.exists():
        pytest.skip("ground_truth.csv not found; run python -m src.generate_data first")
    return pd.read_csv(CFG.GROUND_TRUTH_PATH, dtype=str)


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not CFG.METRICS_PATH.exists():
        pytest.skip("metrics.json not found; run python -m src.generate_data first")
    with open(CFG.METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFileExistence:
    """All expected output files must exist."""

    def test_register_raw_exists(self):
        assert CFG.RAW_REGISTER_PATH.exists(), "register_raw.csv missing"

    def test_ground_truth_exists(self):
        assert CFG.GROUND_TRUTH_PATH.exists(), "ground_truth.csv missing"

    def test_fault_log_exists(self):
        assert CFG.FAULT_LOG_PATH.exists(), "fault_log.csv missing"

    def test_demo_register_exists(self):
        assert CFG.DEMO_REGISTER_PATH.exists(), "demo_register.csv missing"

    def test_sample_batch_exists(self):
        assert CFG.SAMPLE_BATCH_PATH.exists(), "sample_batch.csv missing"


class TestRegisterShape:
    """Register must meet size and column requirements."""

    def test_at_least_2000_records(self, register):
        assert len(register) >= 2_000, f"Only {len(register)} records"

    def test_expected_columns_present(self, register):
        required = {
            "record_id", "source_system", "registration_date",
            "first_name", "family_name", "gender", "date_of_birth",
            "age_years", "national_id", "phone", "city", "address",
            "blood_group", "insurance_tier", "visit_count",
        }
        missing = required - set(register.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_ground_truth_column_in_register(self, register):
        """true_entity_id must NOT appear in the register (leakage prevention)."""
        assert "true_entity_id" not in register.columns

    def test_target_record_count_approximately_50k(self, register):
        """Register should be between 40,000 and 60,000 records."""
        n = len(register)
        assert 40_000 <= n <= 60_000, f"Unexpected record count: {n}"

    def test_at_least_10_attributes(self, register):
        """Brief requires ≥10 attributes."""
        assert len(register.columns) >= 10


class TestGroundTruth:
    """Ground truth file must match the register row count and have no leakage."""

    def test_same_row_count_as_register(self, register, ground_truth):
        assert len(register) == len(ground_truth), (
            f"register has {len(register)} rows, ground_truth has {len(ground_truth)}"
        )

    def test_record_ids_match_exactly(self, register, ground_truth):
        reg_ids = set(register["record_id"].tolist())
        gt_ids  = set(ground_truth["record_id"].tolist())
        assert reg_ids == gt_ids, "Mismatch between register and ground_truth record_ids"

    def test_entity_ids_never_in_register(self, register, ground_truth):
        """No entity ID column should appear anywhere in the register."""
        # The ground truth entity column must not be a column in the register
        assert "true_entity_id" not in register.columns
        # Also verify register column names contain no column that looks like an entity key
        suspicious = [c for c in register.columns if "entity" in c.lower()]
        assert suspicious == [], f"Suspicious entity-like columns in register: {suspicious}"



class TestRecordIdShuffle:
    """record_id must carry no information about entity membership."""

    def test_record_id_unique(self, register):
        assert register["record_id"].is_unique, "record_id values are not unique"

    def test_record_id_not_correlated_with_entity(self, register, ground_truth):
        """Check that record_id sort order does not cluster by entity.

        Method: merge register with ground_truth, sort by record_id, compute
        the fraction of adjacent row-pairs that share the same entity.
        If IDs were NOT shuffled, this fraction would be much higher than
        the expected value under randomness (≈ duplicate_rate).
        A shuffled register should give a fraction close to the random expectation.
        """
        merged = register[["record_id"]].merge(
            ground_truth, on="record_id"
        ).sort_values("record_id").reset_index(drop=True)

        entity_col = merged["true_entity_id"]
        adjacent_same = (entity_col == entity_col.shift(1)).sum()
        fraction_same = adjacent_same / (len(merged) - 1)

        # Under a perfect shuffle, P(same entity at adjacent IDs)
        # ≈ sum_entity (cluster_size * (cluster_size - 1)) / (N * (N-1))
        # which is very small for large N.  We just assert < 5 %.
        assert fraction_same < 0.05, (
            f"Adjacent record_ids share the same entity {fraction_same:.2%} of the time "
            f"— IDs may not be shuffled."
        )


class TestMeasurementTypes:
    """The register must include all four measurement types as per the brief."""

    def test_nominal_attributes_present(self, register):
        nominal_cols = {"record_id", "source_system", "first_name", "family_name",
                        "gender", "national_id", "phone", "city", "blood_group"}
        present = nominal_cols & set(register.columns)
        assert len(present) >= 3, "Expected ≥3 nominal attributes"

    def test_ordinal_attribute_present(self, register):
        assert "insurance_tier" in register.columns, "Ordinal attribute insurance_tier missing"
        valid_tiers = {"None", "Basic", "Standard", "Premium"}
        non_null_values = register["insurance_tier"].dropna().unique()
        # At least some records should have a valid tier
        assert any(v in valid_tiers for v in non_null_values)

    def test_interval_attributes_present(self, register):
        """date_of_birth and registration_date are interval attributes."""
        assert "date_of_birth" in register.columns
        assert "registration_date" in register.columns

    def test_ratio_attributes_present(self, register):
        """age_years and visit_count are ratio attributes (true zero, meaningful ratios)."""
        assert "age_years" in register.columns
        assert "visit_count" in register.columns


class TestSourceSystemConventions:
    """Legacy must have high national_id missing rate; Emergency must have high father_name missing rate."""

    def test_legacy_national_id_mostly_missing(self, register):
        legacy = register[register["source_system"] == "Legacy"]
        if len(legacy) == 0:
            pytest.skip("No Legacy records found")
        missing_rate = legacy["national_id"].isna().mean()
        assert missing_rate > 0.80, (
            f"Legacy national_id missing rate is {missing_rate:.2%}; expected >80%"
        )

    def test_emergency_father_name_often_missing(self, register):
        emergency = register[register["source_system"] == "Emergency"]
        if len(emergency) == 0:
            pytest.skip("No Emergency records found")
        missing_rate = emergency["father_name"].isna().mean()
        assert missing_rate > 0.40, (
            f"Emergency father_name missing rate is {missing_rate:.2%}; expected >40%"
        )


class TestSanityReport:
    """Sanity report in metrics.json must exist and have sane values."""

    def test_generation_key_in_metrics(self, metrics):
        assert "generation" in metrics, "Key 'generation' missing from metrics.json"

    def test_trivial_rule_recall_not_perfect(self, metrics):
        """Trivial rule should NOT have near-perfect recall; ensures data is hard enough."""
        recall = metrics["generation"]["trivial_rule_same_dob_and_phone"]["recall"]
        assert recall < 0.70, (
            f"Trivial rule recall={recall:.3f} is too high (≥0.70). "
            "Data may be too easy — increase corruption probabilities."
        )

    def test_hard_negative_counts_positive(self, metrics):
        hn = metrics["generation"]["hard_negatives"]
        assert hn["sibling_pairs"] > 0
        assert hn["father_son_pairs"] > 0
        assert hn["household_pairs"] > 0

    def test_total_entities_less_than_total_records(self, metrics):
        gen = metrics["generation"]
        assert gen["total_entities"] < gen["total_records"]

    def test_duplicate_rate_in_expected_range(self, metrics):
        rate = metrics["generation"]["duplicate_rate_actual"]
        assert 0.05 <= rate <= 0.25, f"Duplicate rate {rate:.3f} outside expected range [0.05, 0.25]"


class TestSubsetFiles:
    """Demo and sample files must have the right approximate sizes."""

    def test_demo_has_approximately_5000_rows(self):
        if not CFG.DEMO_REGISTER_PATH.exists():
            pytest.skip("demo_register.csv missing")
        demo = pd.read_csv(CFG.DEMO_REGISTER_PATH, dtype=str)
        assert 3_000 <= len(demo) <= 7_000, f"Demo has {len(demo)} rows; expected ~5,000"

    def test_sample_has_approximately_2000_rows(self):
        if not CFG.SAMPLE_BATCH_PATH.exists():
            pytest.skip("sample_batch.csv missing")
        sample = pd.read_csv(CFG.SAMPLE_BATCH_PATH, dtype=str)
        assert 1_000 <= len(sample) <= 3_000, f"Sample has {len(sample)} rows; expected ~2,000"

    def test_demo_no_ground_truth_column(self):
        if not CFG.DEMO_REGISTER_PATH.exists():
            pytest.skip()
        demo = pd.read_csv(CFG.DEMO_REGISTER_PATH, dtype=str)
        assert "true_entity_id" not in demo.columns
