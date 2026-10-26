"""Unit and integration tests for src/features.py, src/split.py, and src/pairs.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as CFG
from src.features import (
    char_ngram_jaccard,
    char_tfidf_cosine,
    check_forbidden_columns,
    compute_pair_features,
    extract_features_for_pair_dataframe,
    token_cosine,
    token_jaccard,
)
from src.pairs import (
    create_stratified_resampled_train_pairs,
    label_pairs,
)
from src.split import compute_temporal_entity_split


# ── Fixtures for Hand-Built Patient Records ────────────────────────────────────

@pytest.fixture
def base_record() -> dict[str, str]:
    """Base template for patient record with all allowlisted attributes."""
    return {
        "record_id": "REC0001",
        "source_system": "OPD",
        "registration_date": "2020-01-15",
        "first_name": "Muhammad",
        "family_name": "Khan",
        "father_name": "Tariq Khan",
        "gender": "M",
        "date_of_birth": "1985-05-12",
        "age_years": "34",
        "national_id": "3520112345671",
        "phone": "03001234567",
        "city": "Lahore",
        "address": "House 14, Street 2, Gulberg III, Lahore",
        "blood_group": "B+",
        "insurance_tier": "Standard",
        "visit_count": "5",
    }


# ── Test Suite 1: Specific Hand-Built Semantic Pairs ───────────────────────────

class TestHandBuiltSemanticPairs:
    """Verifies feature values on specific hand-built record comparisons."""

    def test_muhammad_vs_mohammad_spelling_variant(self, base_record: dict[str, str]) -> None:
        """Muhammad vs Mohammad: high Jaro-Winkler, identical Soundex, non-exact."""
        r1 = base_record.copy()
        r2 = base_record.copy()
        r2["record_id"] = "REC0002"
        r2["first_name"] = "Mohammad"

        feats = compute_pair_features(r1, r2)

        # First name comparisons
        assert feats["fn_exact"] == 0.0
        assert feats["fn_soundex_equal"] == 1.0  # Both map to M530
        assert feats["fn_initial_equal"] == 1.0
        assert feats["fn_jaro_winkler"] > 0.90
        assert feats["fn_levenshtein"] > 0.85
        assert feats["fn_both_present"] == 1.0

        # Full name custom metrics
        assert feats["full_name_ngram_jaccard"] > 0.60
        assert feats["full_name_tfidf_cosine"] > 0.65

        # Other fields remain identical
        assert feats["fam_exact"] == 1.0
        assert feats["dob_exact"] == 1.0
        assert feats["nid_exact"] == 1.0
        assert feats["phone_exact"] == 1.0
        assert feats["gower_similarity"] > 0.95

    def test_swapped_day_and_month(self, base_record: dict[str, str]) -> None:
        """1985-05-12 vs 1985-12-05: dob_day_month_swapped must be 1.0."""
        r1 = base_record.copy()
        r1["date_of_birth"] = "1985-05-12"

        r2 = base_record.copy()
        r2["record_id"] = "REC0003"
        r2["date_of_birth"] = "1985-12-05"

        feats = compute_pair_features(r1, r2)

        assert feats["dob_both_present"] == 1.0
        assert feats["dob_exact"] == 0.0
        assert feats["dob_year_equal"] == 1.0
        assert feats["dob_day_equal"] == 0.0
        assert feats["dob_month_equal"] == 0.0
        assert feats["dob_day_month_swapped"] == 1.0
        assert feats["dob_diff_days_log"] > 0.0

    def test_missing_national_id_on_one_side(self, base_record: dict[str, str]) -> None:
        """Missing national_id on one side: nid_both_present must be 0, Gower delta rule skips NID."""
        r1 = base_record.copy()
        r1["national_id"] = "3520112345671"

        r2 = base_record.copy()
        r2["record_id"] = "REC0004"
        r2["national_id"] = ""  # Emergency / uncollected

        feats = compute_pair_features(r1, r2)

        assert feats["nid_both_present"] == 0.0
        assert feats["nid_exact"] == 0.0
        assert feats["nid_hamming_le_1"] == 0.0
        assert feats["nid_levenshtein"] == 0.0

        # Gower similarity must compute gracefully without dividing by 0 or crashing
        assert 0.0 < feats["gower_similarity"] <= 1.0

    def test_hard_negative_siblings(self, base_record: dict[str, str]) -> None:
        """Siblings: same family name, father name, phone, address, city; different given name and DOB."""
        r1 = base_record.copy()
        r1["first_name"] = "Hamza"
        r1["family_name"] = "Abbasi"
        r1["father_name"] = "Tariq Abbasi"
        r1["date_of_birth"] = "1995-04-10"
        r1["national_id"] = "3520111111111"

        r2 = base_record.copy()
        r2["record_id"] = "REC0005"
        r2["first_name"] = "Bilal"
        r2["family_name"] = "Abbasi"
        r2["father_name"] = "Tariq Abbasi"
        r2["date_of_birth"] = "1998-09-22"
        r2["national_id"] = "3520122222222"

        feats = compute_pair_features(r1, r2)

        # Agreement features
        assert feats["fam_exact"] == 1.0
        assert feats["father_exact"] == 1.0
        assert feats["phone_exact"] == 1.0
        assert feats["city_equal"] == 1.0
        assert feats["address_token_jaccard"] == 1.0

        # Disagreement features (preventing false duplicate match)
        assert feats["fn_exact"] == 0.0
        assert feats["fn_soundex_equal"] == 0.0  # H520 vs B440
        assert feats["fn_jaro_winkler"] < 0.60
        assert feats["dob_exact"] == 0.0
        assert feats["dob_year_equal"] == 0.0
        assert feats["nid_exact"] == 0.0
        assert feats["nid_hamming_le_1"] == 0.0

    def test_hard_negative_father_and_son(self, base_record: dict[str, str]) -> None:
        """Father and son: exact same full name and address, but ~30 years DOB gap."""
        r1 = base_record.copy()
        r1["first_name"] = "Muhammad"
        r1["family_name"] = "Aslam"
        r1["date_of_birth"] = "1960-01-15"
        r1["age_years"] = "59"
        r1["national_id"] = "3520155555555"

        r2 = base_record.copy()
        r2["record_id"] = "REC0006"
        r2["first_name"] = "Muhammad"
        r2["family_name"] = "Aslam"
        r2["date_of_birth"] = "1990-01-15"
        r2["age_years"] = "29"
        r2["national_id"] = "3520199999999"

        feats = compute_pair_features(r1, r2)

        # Exact name agreement
        assert feats["fn_exact"] == 1.0
        assert feats["fam_exact"] == 1.0
        assert feats["full_name_jaro_winkler"] == 1.0
        assert feats["full_name_levenshtein"] == 1.0

        # Clear generational gap
        assert feats["dob_exact"] == 0.0
        assert feats["dob_year_equal"] == 0.0
        assert feats["dob_diff_days_log"] > 8.0  # 30 years ~ 10,950 days -> ln(10951) ~ 9.3
        assert feats["age_diff_abs"] >= 29.0
        assert feats["nid_exact"] == 0.0

    def test_name_swap_flag(self, base_record: dict[str, str]) -> None:
        """First and family names inverted: name_swap_flag must be 1.0."""
        r1 = base_record.copy()
        r1["first_name"] = "Tariq"
        r1["family_name"] = "Khan"

        r2 = base_record.copy()
        r2["record_id"] = "REC0007"
        r2["first_name"] = "Khan"
        r2["family_name"] = "Tariq"

        feats = compute_pair_features(r1, r2)
        assert feats["name_swap_flag"] == 1.0


# ── Test Suite 2: Custom String Metrics Logic ─────────────────────────────────

class TestCustomStringMetrics:
    """Verifies self-implemented character n-gram Jaccard and TF-IDF Cosine."""

    def test_char_ngram_jaccard_properties(self) -> None:
        """Tests reflexivity, bounds, and sensitivity."""
        assert char_ngram_jaccard("abc", "abc") == 1.0
        assert char_ngram_jaccard("", "abc") == 0.0
        assert char_ngram_jaccard("abc", "") == 0.0
        assert char_ngram_jaccard("ab", "cd") == 0.0

        # Partial overlap: "muhammad" vs "mohammad"
        j = char_ngram_jaccard("muhammad", "mohammad", n=2)
        assert 0.50 < j < 0.80

    def test_char_tfidf_cosine_properties(self) -> None:
        """Tests reflexivity, bounds, and sensitivity of TF-IDF cosine."""
        assert char_tfidf_cosine("abcde", "abcde") == 1.0
        assert char_tfidf_cosine("", "abcde") == 0.0
        assert char_tfidf_cosine("abcde", "xyzuv") == 0.0

        c = char_tfidf_cosine("muhammad khan", "mohammad khaan", n=3)
        assert 0.50 < c < 0.90

    def test_token_jaccard_and_cosine(self) -> None:
        """Tests word-token similarities for address strings."""
        a1 = "House 10, Street 5, Gulberg, Lahore"
        a2 = "House 10, Street 5, Gulberg III, Lahore"

        tj = token_jaccard(a1, a2)
        tc = token_cosine(a1, a2)
        assert tj > 0.70
        assert tc > 0.70


# ── Test Suite 3: Forbidden-Column & Allowlist Guards ──────────────────────────

class TestLeakageAndAllowlistGuards:
    """Ensures prohibited columns raise ValueError immediately upon detection."""

    def test_check_forbidden_columns_raises_on_true_entity_id(self) -> None:
        """Presence of true_entity_id must raise ValueError."""
        cols = ["record_id", "first_name", "true_entity_id"]
        with pytest.raises(ValueError, match="FORBIDDEN COLUMN GUARD"):
            check_forbidden_columns(cols)

    def test_check_forbidden_columns_raises_on_fault_type(self) -> None:
        """Presence of generator metadata must raise ValueError."""
        cols = ["record_id", "first_name", "fault_type"]
        with pytest.raises(ValueError, match="FORBIDDEN COLUMN GUARD"):
            check_forbidden_columns(cols)

    def test_check_forbidden_columns_raises_on_unallowlisted_columns(self) -> None:
        """Presence of arbitrary unallowlisted columns must raise ValueError."""
        cols = ["record_id", "first_name", "unknown_secret_column"]
        with pytest.raises(ValueError, match="ALLOWLIST GUARD"):
            check_forbidden_columns(cols)

    def test_check_forbidden_columns_passes_on_clean_allowlist(self) -> None:
        """Clean model allowlist columns must pass without error."""
        check_forbidden_columns(CFG.CLEAN_MODEL_ALLOWLIST)

    def test_extract_features_dataframe_raises_on_forbidden_column(self) -> None:
        """Passing a DataFrame with true_entity_id into extract_features raises ValueError."""
        df_dirty = pd.DataFrame([{
            "record_id": "REC0001",
            "first_name": "Ali",
            "family_name": "Khan",
            "true_entity_id": "ENT0001",  # Leakage!
        }])
        pairs_df = pd.DataFrame([{"record_id_1": "REC0001", "record_id_2": "REC0001"}])

        with pytest.raises(ValueError, match="FORBIDDEN COLUMN GUARD"):
            extract_features_for_pair_dataframe(pairs_df, df_dirty)


# ── Test Suite 4: Splitting & Pairs Generation Modules ─────────────────────────

class TestSplittingAndPairs:
    """Verifies temporal splitting assertions and pairs labelling."""

    def test_entity_splits_json_exists_and_valid(self) -> None:
        """Checks entity_splits.json integrity."""
        split_meta_path = CFG.DATA_PAIRS_DIR / "entity_splits.json"
        assert split_meta_path.exists(), f"Missing {split_meta_path}"

        with open(split_meta_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "cutoff_date" in data
        assert data["train_entities_count"] > 0
        assert data["test_entities_count"] > 0
        # Check roughly 70% train
        assert 0.65 <= data["train_entity_ratio"] <= 0.75

        # Leakage check: zero entity overlap
        train_ents = set(k for k, v in data["entity_to_side"].items() if v == "train")
        test_ents = set(k for k, v in data["entity_to_side"].items() if v == "test")
        assert len(train_ents & test_ents) == 0

    def test_label_pairs_unit(self) -> None:
        """Verifies candidate pair labelling with ground truth map."""
        gt_map = {
            "R1": "ENT_A",
            "R2": "ENT_A",  # Same entity
            "R3": "ENT_B",  # Different entity
        }
        pairs = {("R1", "R2"), ("R1", "R3"), ("R2", "R3")}
        df_labelled = label_pairs(pairs, gt_map)

        row_12 = df_labelled[(df_labelled["record_id_1"] == "R1") & (df_labelled["record_id_2"] == "R2")]
        row_13 = df_labelled[(df_labelled["record_id_1"] == "R1") & (df_labelled["record_id_2"] == "R3")]

        assert len(row_12) == 1
        assert row_12.iloc[0]["label"] == 1
        assert len(row_13) == 1
        assert row_13.iloc[0]["label"] == 0

    def test_resampled_train_pairs_weights_calculation(self) -> None:
        """Tests that stratified resampling keeps all positives and attaches correct weights."""
        df_mock = pd.DataFrame({
            "record_id_1": [f"R{i}" for i in range(100)],
            "record_id_2": [f"R{i+100}" for i in range(100)],
            "label": [1] * 10 + [0] * 90,  # 10 positives, 90 negatives
        })

        resampled, w_pos, w_neg = create_stratified_resampled_train_pairs(
            df_mock, negative_ratio=2.0, seed=42
        )

        # Target negatives = 10 * 2 = 20
        assert len(resampled[resampled["label"] == 1]) == 10
        assert len(resampled[resampled["label"] == 0]) == 20
        assert len(resampled) == 30

        assert w_pos == 1.0
        assert w_neg == 90.0 / 20.0  # 4.5
        assert (resampled[resampled["label"] == 1]["sample_weight"] == 1.0).all()
        assert (resampled[resampled["label"] == 0]["sample_weight"] == 4.5).all()


# ── Test Suite 5: Artifacts and Saved Feature Files ────────────────────────────

class TestSavedFeatureArtifacts:
    """Verifies that generated feature datasets and documentation exist and are valid."""

    def test_feature_dictionary_md_exists_and_covers_all_features(self) -> None:
        """Verifies docs/feature_dictionary.md exists and documents key features."""
        dict_path = CFG.DOCS_DIR / "feature_dictionary.md"
        assert dict_path.exists(), f"Missing {dict_path}"

        content = dict_path.read_text(encoding="utf-8")
        assert "fn_jaro_winkler" in content
        assert "dob_day_month_swapped" in content
        assert "gower_similarity" in content
        assert "name_swap_flag" in content
        assert "nid_hamming_le_1" in content
        assert "Missing Value Policy by Model Family" in content

    def test_feature_data_files_exist_and_non_empty(self) -> None:
        """Verifies generated feature files exist in data/pairs/."""
        for fname in [
            "train_features_resampled.parquet",
            "test_features.parquet",
            "train_pairs_natural.parquet",
            "test_pairs.parquet",
        ]:
            fpath = CFG.DATA_PAIRS_DIR / fname
            assert fpath.exists(), f"Missing pair file: {fpath}"
            assert fpath.stat().st_size > 1000, f"File {fpath} is suspiciously small"

    def test_feature_matrix_shapes_and_columns(self) -> None:
        """Verifies train and test feature matrices have correct columns and no NaN labels."""
        train_df = pd.read_parquet(CFG.DATA_PAIRS_DIR / "train_features_resampled.parquet")
        test_df = pd.read_parquet(CFG.DATA_PAIRS_DIR / "test_features.parquet")

        assert "label" in train_df.columns
        assert "sample_weight" in train_df.columns
        assert "label" in test_df.columns

        # Check total columns (at least 60: metadata + 58 comparison features)
        assert train_df.shape[1] >= 60
        assert test_df.shape[1] >= 60

        assert train_df["label"].isna().sum() == 0
        assert test_df["label"].isna().sum() == 0
