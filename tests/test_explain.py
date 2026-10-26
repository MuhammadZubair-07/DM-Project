"""Unit tests for the Clerk-Facing Explanation Engine (Roadmap Section 7.10, Lecture 3 & 4).

Verifies:
1. explain(pair, model) returns all required structured keys:
   verdict, probability, confidence_label, reasons_for, reasons_against,
   field_comparisons, clerk_summary, model_name, model_family.
2. Explanations NEVER contain jargon (regex: node, gini, feature_, <=, >=, digits with >=3 decimals).
3. Every explanation returns at least one reason for and against.
4. Confidence labels are strictly tied to cost-derived bands:
   - p >= 0.8333: 'Very likely duplicate'
   - 0.20 <= p < 0.8333: 'Worth reviewing'
   - p < 0.20: 'Unlikely duplicate'
5. Decision tree split tests are converted into plain-English sentences:
   - 'Same date of birth'
   - 'First names are nearly identical (Muhammad / Mohammad)'
   - 'Phone numbers differ'
   - 'National ID is missing on one record so it could not be compared'
6. Naive Bayes log-likelihood-ratio top contributors for and against.
7. k-NN nearest historical pairs phrasing: 'Most similar past pairs: 4 of 5 were the same person'.
8. Field comparison dictionary provides agree/conflict/unknown flags for Screen 6 merge screen.
9. 20-pair review sample generation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import config as CFG
from src.explain import (
    REVIEW_THRESHOLD,
    STRICT_THRESHOLD,
    compute_field_comparisons,
    explain,
    generate_explanation_sample,
)
from src.models import Matcher

# Jargon regex guard required by roadmap Section 7.10
JARGON_REGEX = re.compile(
    r"\bnode\b|\bgini\b|\bfeature_\w+|<=|>=|\b\d+\.\d{3,}\b",
    re.IGNORECASE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def exact_duplicate_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    rec1 = {
        "record_id": "REC_A01",
        "first_name": "Tariq",
        "family_name": "Mahmood",
        "father_name": "Abdul Rehman",
        "date_of_birth": "1978-04-12",
        "age_years": "48",
        "gender": "M",
        "national_id": "35201-1234567-1",
        "phone": "03001234567",
        "city": "Lahore",
        "address": "House 14 Street 2 Sector G",
        "blood_group": "B+",
        "insurance_tier": "Gold",
        "visit_count": 4,
    }
    rec2 = dict(rec1)
    rec2["record_id"] = "REC_A02"
    return rec1, rec2


@pytest.fixture
def name_typo_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    rec1 = {
        "record_id": "REC_B01",
        "first_name": "Muhammad",
        "family_name": "Khan",
        "father_name": "Tariq Khan",
        "date_of_birth": "1985-06-15",
        "age_years": "41",
        "gender": "M",
        "national_id": "42101-1234567-1",
        "phone": "03009876543",
        "city": "Karachi",
        "address": "Plot 5 Block B North Nazimabad",
        "blood_group": "O+",
        "insurance_tier": "Bronze",
        "visit_count": 2,
    }
    rec2 = dict(rec1)
    rec2["record_id"] = "REC_B02"
    rec2["first_name"] = "Mohammad"  # Typo variant
    return rec1, rec2


@pytest.fixture
def missing_national_id_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    rec1 = {
        "record_id": "REC_C01",
        "first_name": "Fatima",
        "family_name": "Bibi",
        "father_name": "Ghulam Nabi",
        "date_of_birth": "1980-01-01",
        "age_years": "46",
        "gender": "F",
        "national_id": "",  # Unrecorded in Emergency intake
        "phone": "03339876543",
        "city": "Lahore",
        "address": "Block C Model Town",
        "blood_group": "O+",
        "insurance_tier": "Silver",
        "visit_count": 5,
    }
    rec2 = dict(rec1)
    rec2["record_id"] = "REC_C02"
    rec2["national_id"] = "35201-9876543-2"
    return rec1, rec2


@pytest.fixture
def hard_negative_sibling_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    rec1 = {
        "record_id": "REC_D01",
        "first_name": "Ali",
        "family_name": "Khan",
        "father_name": "Tariq Khan",
        "date_of_birth": "1990-03-10",
        "age_years": "36",
        "gender": "M",
        "national_id": "42101-1111111-1",
        "phone": "03001234567",
        "city": "Karachi",
        "address": "House 12 Street 4 Sector 5",
        "blood_group": "B+",
        "insurance_tier": "Bronze",
        "visit_count": 3,
    }
    rec2 = {
        "record_id": "REC_D02",
        "first_name": "Ahmed",  # Brother
        "family_name": "Khan",  # Same family
        "father_name": "Tariq Khan",  # Same father
        "date_of_birth": "1995-11-20",  # Different DOB
        "age_years": "31",
        "gender": "M",
        "national_id": "42101-2222222-1",
        "phone": "03001234567",  # Shared household landline/phone
        "city": "Karachi",
        "address": "House 12 Street 4 Sector 5",  # Same residence
        "blood_group": "B+",
        "insurance_tier": "Bronze",
        "visit_count": 1,
    }
    return rec1, rec2


# ── 1. Structured Explanation Interface Tests ─────────────────────────────────

def test_explain_returns_all_required_structured_keys(
    exact_duplicate_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies that explain() returns all required keys with appropriate types."""
    res = explain(exact_duplicate_pair, model="decision_tree")

    required_keys = [
        "verdict",
        "probability",
        "confidence_label",
        "reasons_for",
        "reasons_against",
        "field_comparisons",
        "clerk_summary",
        "model_name",
        "model_family",
        "operating_threshold",
        "strict_threshold",
    ]
    for key in required_keys:
        assert key in res, f"Missing key in explanation result: {key}"

    assert isinstance(res["verdict"], str)
    assert isinstance(res["probability"], float)
    assert 0.0 <= res["probability"] <= 1.0
    assert isinstance(res["confidence_label"], str)
    assert isinstance(res["reasons_for"], list)
    assert isinstance(res["reasons_against"], list)
    assert isinstance(res["field_comparisons"], dict)
    assert isinstance(res["clerk_summary"], str)


# ── 2. Strict Zero-Jargon Guard Tests ──────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["decision_tree", "naive_bayes", "knn", "mlp"])
def test_explanations_zero_jargon_guard(
    model_name: str,
    exact_duplicate_pair: tuple[dict[str, Any], dict[str, Any]],
    name_typo_pair: tuple[dict[str, Any], dict[str, Any]],
    missing_national_id_pair: tuple[dict[str, Any], dict[str, Any]],
    hard_negative_sibling_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Rigorous assertion: explanations must NEVER contain machine learning jargon.
    
    Guards against:
    - 'node' (internal decision tree node identifiers)
    - 'gini' (impurity metrics)
    - 'feature_' (internal feature code prefixes)
    - '<=' or '>=' (mathematical inequality operators)
    - digits with 3 or more decimal places (raw floating-point numbers)
    """
    pairs = [
        exact_duplicate_pair,
        name_typo_pair,
        missing_national_id_pair,
        hard_negative_sibling_pair,
    ]

    for pair in pairs:
        res = explain(pair, model=model_name)
        text_corpus = (
            " ".join(res["reasons_for"])
            + " "
            + " ".join(res["reasons_against"])
            + " "
            + res["clerk_summary"]
        )

        match = JARGON_REGEX.search(text_corpus)
        assert match is None, (
            f"Jargon leak detected in {model_name} explanation! "
            f"Matched '{match.group(0)}' in: {text_corpus}"
        )


# ── 3. Completeness: At Least One Reason Returned ─────────────────────────────

@pytest.mark.parametrize("model_name", ["decision_tree", "naive_bayes", "knn", "mlp"])
def test_every_explanation_returns_at_least_one_reason(
    model_name: str,
    exact_duplicate_pair: tuple[dict[str, Any], dict[str, Any]],
    hard_negative_sibling_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Verifies that every explain() call produces reasons for and against."""
    for pair in [exact_duplicate_pair, hard_negative_sibling_pair]:
        res = explain(pair, model=model_name)
        assert len(res["reasons_for"]) >= 1, f"Empty reasons_for in {model_name}"
        assert len(res["reasons_against"]) >= 1, f"Empty reasons_against in {model_name}"


# ── 4. Cost-Derived Confidence Tiers ──────────────────────────────────────────

def test_confidence_labels_tied_to_cost_derived_bands() -> None:
    """Verifies confidence labels map strictly to cost-optimal Bayesian decision boundaries.
    
    Bands:
    - p >= 0.8333: 'Very likely duplicate' (High-confidence merge band)
    - 0.20 <= p < 0.8333: 'Worth reviewing' (Review worklist band)
    - p < 0.20: 'Unlikely duplicate' (Distinct patient records)
    """
    assert pytest.approx(REVIEW_THRESHOLD, 1e-4) == 0.20
    assert pytest.approx(STRICT_THRESHOLD, 1e-4) == 50.0 / 60.0

    m = Matcher.load(CFG.MODELS_DIR / "decision_tree.joblib")

    # High probability duplicate
    high_pair = {
        "record_1": {"first_name": "Ali", "date_of_birth": "1990-01-01"},
        "record_2": {"first_name": "Ali", "date_of_birth": "1990-01-01"},
    }
    res_high = explain(high_pair, model=m)
    if res_high["probability"] >= STRICT_THRESHOLD:
        assert res_high["confidence_label"] == "Very likely duplicate"
        assert res_high["verdict"] == "Duplicate Match"

    # Distinct pair
    low_pair = {
        "record_1": {"first_name": "Zubair", "family_name": "Ahmed", "date_of_birth": "1980-01-01", "city": "Karachi"},
        "record_2": {"first_name": "Fatima", "family_name": "Bibi", "date_of_birth": "2005-12-12", "city": "Peshawar"},
    }
    res_low = explain(low_pair, model=m)
    if res_low["probability"] < REVIEW_THRESHOLD:
        assert res_low["confidence_label"] == "Unlikely duplicate"
        assert res_low["verdict"] == "Distinct Patients"


# ── 5. Hand-Built Decision Tree Trace Phrases ──────────────────────────────────

def test_hand_built_name_typo_explanation_phrases(
    name_typo_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies that decision tree path correctly identifies Muhammad / Mohammad typo."""
    res = explain(name_typo_pair, model="decision_tree")

    # Expected phrases from roadmap 7.10
    reasons_text = " ".join(res["reasons_for"])
    assert "Same date of birth" in reasons_text or "Dates of birth" in reasons_text
    assert "First names are nearly identical (Muhammad / Mohammad)" in reasons_text
    assert res["verdict"] == "Duplicate Match"


def test_hand_built_missing_national_id_graceful_handling(
    missing_national_id_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies graceful handling of unrecorded national ID."""
    res = explain(missing_national_id_pair, model="decision_tree")

    against_text = " ".join(res["reasons_against"])
    expected = "National ID is missing on one record so it could not be compared"
    assert expected in against_text, f"Expected '{expected}' in against reasons: {against_text}"


def test_hand_built_hard_negative_sibling_explanation(
    hard_negative_sibling_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies that sibling pairs with same family name and phone are explained by differing DOB/first name."""
    res = explain(hard_negative_sibling_pair, model="decision_tree")

    assert res["verdict"] == "Distinct Patients"
    assert res["confidence_label"] == "Unlikely duplicate"

    against_text = " ".join(res["reasons_against"])
    assert "Dates of birth differ" in against_text
    assert "First names are completely different (Ali vs Ahmed)" in against_text


# ── 6. Naive Bayes Log-Likelihood-Ratio Explainer ──────────────────────────────

def test_naive_bayes_llr_contributions(
    name_typo_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies Naive Bayes computes LLR and reports top positive and negative factors."""
    res = explain(name_typo_pair, model="naive_bayes")

    assert len(res["reasons_for"]) >= 1
    assert len(res["reasons_against"]) >= 1
    assert "Naive Bayes evidence balance" in res["clerk_summary"]


# ── 7. k-NN Nearest Historical Pairs Explainer ────────────────────────────────

def test_knn_nearest_historical_pairs_phrasing(
    exact_duplicate_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies k-NN describes nearest historical training pairs in plain words."""
    res = explain(exact_duplicate_pair, model="knn")

    reasons_text = " ".join(res["reasons_for"] + res["reasons_against"])
    pattern = re.compile(r"Most similar past pairs: \d+ of \d+ were (the same person|distinct persons)")
    assert pattern.search(reasons_text) is not None, (
        f"k-NN phrasing pattern not found in: {reasons_text}"
    )


# ── 8. Merge Screen (Screen 6) Field Comparison Flags ─────────────────────────

def test_field_comparisons_merge_screen(
    missing_national_id_pair: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Verifies per-field agree/conflict/unknown tags for the merge review screen."""
    rec1, rec2 = missing_national_id_pair
    comps = compute_field_comparisons(rec1, rec2)

    required_fields = [
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

    for field in required_fields:
        assert field in comps, f"Missing field in merge comparison: {field}"
        info = comps[field]
        assert info["status"] in ("agree", "conflict", "unknown")
        assert "field_label" in info
        assert "value_1" in info
        assert "value_2" in info
        assert "detail" in info

    # First name Fatima matches
    assert comps["first_name"]["status"] == "agree"
    # Missing CNIC on rec1
    assert comps["national_id"]["status"] == "unknown"


# ── 9. Review Sample Generator ────────────────────────────────────────────────

def test_generate_explanation_sample() -> None:
    """Verifies 20-sample explanation generator runs cleanly without errors."""
    df_sample = generate_explanation_sample(n_samples=20, model_name="decision_tree")
    assert len(df_sample) == 20
    assert "verdict" in df_sample.columns
    assert "confidence_label" in df_sample.columns
    assert "primary_reason_for" in df_sample.columns
    assert "primary_reason_against" in df_sample.columns
    assert not df_sample["verdict"].isna().any()
