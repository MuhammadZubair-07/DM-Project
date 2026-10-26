"""Unit and integration tests for src/blocking.py (Candidate Blocking & Assigned Challenge)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config as CFG
from src.blocking import (
    calculate_blocking_metrics,
    pass_b1_soundex_dob_year,
    pass_b2_fn3_dob_year_month,
    pass_b3_phone_last7,
    pass_b4_national_id_exact,
    pass_b5_sorted_neighbourhood,
    soundex,
)


class TestSoundex:
    """Tests for pure-Python Soundex implementation."""

    def test_known_phonetic_encodings(self) -> None:
        """Verifies standard phonetic soundex mappings."""
        assert soundex("Khan") == "K500"
        assert soundex("Ahmed") == "A530"
        assert soundex("Ali") == "A400"
        assert soundex("Robert") == "R163"
        assert soundex("Rupert") == "R163"
        assert soundex("Smith") == "S530"
        assert soundex("Smyth") == "S530"
        assert soundex("Pfister") == "P236"
        assert soundex("Honeyman") == "H555"

    def test_edge_cases(self) -> None:
        """Verifies handling of empty, whitespace, and dirty strings."""
        assert soundex("") == ""
        assert soundex("   ") == ""
        assert soundex("12345") == ""
        assert soundex("  Khan 123! ") == "K500"
        # Adjacent identical phonetic codes collapsed
        assert soundex("Jackson") == "J250"
        # Pure vowels after initial letter
        assert soundex("Aeo") == "A000"


class TestHandbuiltBlockingPasses:
    """Tests individual passes and union on small handbuilt registers with known answers."""

    @pytest.fixture
    def tiny_register(self) -> pd.DataFrame:
        """Creates a controlled 6-record register with known overlaps."""
        return pd.DataFrame([
            {
                "record_id": "REC001",
                "first_name": "Muhammad",
                "family_name": "Khan",
                "date_of_birth": "1990-05-15",
                "phone": "03001234567",
                "national_id": "3520112345671",
            },
            {
                "record_id": "REC002",
                "first_name": "Muhammad",
                "family_name": "Khan",
                "date_of_birth": "1990-05-20",
                "phone": "03009999999",
                "national_id": "3520199999999",
            },
            {
                "record_id": "REC003",
                "first_name": "Zubair",
                "family_name": "Khan",
                "date_of_birth": "1990-12-01",
                "phone": "03001234567",
                "national_id": "3520199999998",
            },
            {
                "record_id": "REC004",
                "first_name": "Fatima",
                "family_name": "Malik",
                "date_of_birth": "1985-08-10",
                "phone": "03215555555",
                "national_id": "3520112345671",
            },
            {
                "record_id": "REC005",
                "first_name": "Tariq",
                "family_name": "Abbasi",
                "date_of_birth": "2000-01-01",
                "phone": "03451111111",
                "national_id": "3520177777777",
            },
            {
                "record_id": "REC006",
                "first_name": "Tariq",
                "family_name": "Abbasi",
                "date_of_birth": "2000-02-02",
                "phone": "03452222222",
                "national_id": "3520166666666",
            },
        ])

    def test_b1_soundex_dob_year(self, tiny_register: pd.DataFrame) -> None:
        """B1 blocks on Soundex(family_name) + YYYY."""
        pairs, blocks, dropped_b, dropped_p = pass_b1_soundex_dob_year(tiny_register, max_block_size=10)
        assert dropped_b == 0
        assert dropped_p == 0
        # REC001, REC002, REC003 all have Soundex 'K500' and DOB year '1990'
        expected_k500 = {("REC001", "REC002"), ("REC001", "REC003"), ("REC002", "REC003")}
        # REC005 and REC006 both have Soundex 'A120' and DOB year '2000'
        expected_a120 = {("REC005", "REC006")}
        assert expected_k500.issubset(pairs)
        assert expected_a120.issubset(pairs)
        assert len(pairs) == 4

    def test_b2_fn3_dob_year_month(self, tiny_register: pd.DataFrame) -> None:
        """B2 blocks on first 3 letters of first_name + YYYY-MM."""
        pairs, blocks, dropped_b, dropped_p = pass_b2_fn3_dob_year_month(tiny_register, max_block_size=10)
        assert dropped_b == 0
        # REC001 and REC002 both have first_name[:3] == 'muh' and DOB[:7] == '1990-05'
        assert ("REC001", "REC002") in pairs
        # REC003 has 1990-12, REC005 has 2000-01, REC006 has 2000-02 -> no match
        assert len(pairs) == 1

    def test_b3_phone_last7(self, tiny_register: pd.DataFrame) -> None:
        """B3 blocks on last 7 digits of phone."""
        pairs, blocks, dropped_b, dropped_p = pass_b3_phone_last7(tiny_register, max_block_size=10)
        assert dropped_b == 0
        # REC001 and REC003 both share phone last 7 digits '1234567'
        assert ("REC001", "REC003") in pairs
        assert len(pairs) == 1

    def test_b4_national_id_exact(self, tiny_register: pd.DataFrame) -> None:
        """B4 blocks on exact 13-digit national_id."""
        pairs, blocks, dropped_b, dropped_p = pass_b4_national_id_exact(tiny_register, max_block_size=10)
        assert dropped_b == 0
        # REC001 and REC004 both have national_id '3520112345671'
        assert ("REC001", "REC004") in pairs
        assert len(pairs) == 1

    def test_b5_sorted_neighbourhood(self, tiny_register: pd.DataFrame) -> None:
        """B5 generates sliding window comparisons on compound name."""
        pairs, n_records = pass_b5_sorted_neighbourhood(tiny_register, window_size=2)
        assert n_records == 6
        # Adjacent names in alphabetical sort should be paired
        assert len(pairs) > 0
        # Pairs must be sorted tuples
        for r1, r2 in pairs:
            assert r1 < r2


class TestMissingKeyAndOversizedBlocks:
    """Tests edge cases: missing keys and oversized block suppression."""

    def test_missing_keys_generate_no_blocks(self) -> None:
        """Records with empty/null keys must generate no block for that pass."""
        df_empty = pd.DataFrame([
            {
                "record_id": "REC_BLANK_1",
                "first_name": "",
                "family_name": "",
                "date_of_birth": "",
                "phone": "",
                "national_id": "",
            },
            {
                "record_id": "REC_BLANK_2",
                "first_name": "  ",
                "family_name": "",
                "date_of_birth": "",
                "phone": "0300",  # Short phone (< 7 digits)
                "national_id": "123",  # Short NID (< 13 digits)
            },
        ])

        p1, b1, _, _ = pass_b1_soundex_dob_year(df_empty)
        p2, b2, _, _ = pass_b2_fn3_dob_year_month(df_empty)
        p3, b3, _, _ = pass_b3_phone_last7(df_empty)
        p4, b4, _, _ = pass_b4_national_id_exact(df_empty)
        p5, _ = pass_b5_sorted_neighbourhood(df_empty, window_size=5)

        assert len(p1) == 0
        assert len(p2) == 0
        assert len(p3) == 0
        assert len(p4) == 0
        assert len(p5) == 0

    def test_oversized_block_guard_skips_and_records_dropped(self) -> None:
        """Blocks exceeding max_block_size must be skipped, with dropped pairs logged."""
        # Create 10 records with identical family_name and birth year
        rows = []
        for i in range(10):
            rows.append({
                "record_id": f"REC_BIG_{i:03d}",
                "first_name": "Ali",
                "family_name": "Khan",
                "date_of_birth": "1990-01-01",
                "phone": f"0300{i:07d}",
                "national_id": f"35201{i:08d}",
            })
        df_giant = pd.DataFrame(rows)

        # 10 records in one block would normally yield 10*9/2 = 45 pairs.
        # Set max_block_size=5 -> block size 10 > 5 -> must be dropped!
        pairs, blocks, dropped_blocks, dropped_pairs_est = pass_b1_soundex_dob_year(
            df_giant, max_block_size=5
        )

        assert len(pairs) == 0
        assert dropped_blocks == 1
        assert dropped_pairs_est == 45

        # With generous max_block_size=20, block should NOT be dropped
        pairs_kept, _, dropped_blocks_0, dropped_pairs_0 = pass_b1_soundex_dob_year(
            df_giant, max_block_size=20
        )
        assert len(pairs_kept) == 45
        assert dropped_blocks_0 == 0
        assert dropped_pairs_0 == 0


class TestEvaluationMetrics:
    """Tests mathematical correctness of reduction ratio, completeness, and quality."""

    def test_metrics_calculation_standard(self) -> None:
        """Verifies formulas against manual calculation."""
        candidates = {("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")}
        ground_truth = {("A", "B"), ("C", "D"), ("X", "Y")}  # 3 true pairs, 2 found
        total_records = 10  # all-pairs space = 10 * 9 / 2 = 45

        m = calculate_blocking_metrics(candidates, ground_truth, total_records, runtime_sec=0.5)

        assert m["all_pairs_count"] == 45
        assert m["candidate_pairs_count"] == 4
        assert m["true_pairs_found"] == 2
        assert m["total_true_pairs"] == 3
        # Completeness = 2 / 3 = 0.666667
        assert abs(m["pairs_completeness"] - 2 / 3) < 1e-5
        # Quality = 2 / 4 = 0.5
        assert m["pairs_quality"] == 0.5
        # Reduction ratio = 1 - 4/45 = 41/45 = 0.911111
        assert abs(m["reduction_ratio"] - (1.0 - 4 / 45)) < 1e-5
        assert m["runtime_seconds"] == 0.5

    def test_metrics_zero_candidates(self) -> None:
        """Verifies division-by-zero protection when candidate set is empty."""
        m = calculate_blocking_metrics(set(), {("A", "B")}, 10, 0.0)
        assert m["candidate_pairs_count"] == 0
        assert m["pairs_completeness"] == 0.0
        assert m["pairs_quality"] == 0.0
        assert m["reduction_ratio"] == 1.0


class TestBlockingArtifactsAndOutputs:
    """Validates generated artifacts and gate requirements."""

    def test_blocking_json_exists_and_meets_target(self) -> None:
        """Verifies reports/blocking.json exists and meets pre-registered gate requirements."""
        json_path = CFG.REPORTS_DIR / "blocking.json"
        assert json_path.exists(), f"Missing {json_path}"

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset_name"] == "register_clean.csv"
        assert data["total_records"] == 54373
        assert data["all_pairs_comparisons"] == 1478184378
        assert data["pre_registered_target_completeness_pct"] == 92.0

        schemes = data["schemes_evaluation"]
        assert "UNION (B1-B5)" in schemes
        union = schemes["UNION (B1-B5)"]

        # Gate Requirement: pairs completeness >= 92.0%
        assert union["pairs_completeness_pct"] >= 92.0
        # Hard Ceiling: candidate comparisons <= 1,000,000
        assert union["candidate_pairs_count"] <= 1000000
        # Reduction ratio >= 99.9%
        assert union["reduction_ratio_pct"] >= 99.9

    def test_blocking_report_markdown_exists(self) -> None:
        """Verifies docs/blocking_report.md exists and contains required sections."""
        report_path = CFG.DOCS_DIR / "blocking_report.md"
        assert report_path.exists(), f"Missing {report_path}"

        content = report_path.read_text(encoding="utf-8")
        assert "## 1. Executive Summary & Assigned Challenge Objective" in content
        assert "## 2. Scheme Evaluation & Performance Matrix" in content
        assert "## 3. Empirical Scaling Experiment" in content
        assert "## 4. Missed-Duplicate Analysis" in content
        assert "## 5. What Blocking Costs Us in Missed Duplicates" in content
        assert "## 6. Block-Size Distribution" in content
        assert "## 7. Viva Defence & Theoretical Justification" in content

    def test_figures_exist(self) -> None:
        """Verifies generated figures exist and are non-empty."""
        for fig_name in [
            "blocking_scaling.png",
            "block_size_distribution.png",
            "blocking_tradeoff.png",
        ]:
            fig_path = CFG.FIGURES_DIR / fig_name
            assert fig_path.exists(), f"Missing figure {fig_path}"
            assert fig_path.stat().st_size > 1000, f"Figure {fig_path} is suspiciously small"

    def test_metrics_json_has_blocking_key(self) -> None:
        """Verifies reports/metrics.json contains 'blocking' key."""
        metrics_path = CFG.METRICS_PATH
        assert metrics_path.exists(), f"Missing {metrics_path}"

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        assert "blocking" in metrics
        b = metrics["blocking"]
        assert b["target_completeness_met"] is True
        assert b["pairs_completeness_pct"] >= 92.0
        assert b["union_candidate_pairs"] <= 1000000
