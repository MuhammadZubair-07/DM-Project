"""Unit and integration tests for trained models, Matcher interface, and model selection artifacts.

Verifies:
1. Models load cleanly from disk via Matcher.load.
2. Models produce valid probabilities in [0, 1] summing to 1.0.
3. Predictions are deterministic for fixed inputs.
4. Matcher interface methods (fit, predict_proba, predict, explain, save, load) function as specified.
5. Zero-probability demonstration proves Laplace smoothing eliminates catastrophic false negatives.
6. Reports and metadata artifacts (reports/models.json, docs/model_selection.md) exist and are valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as CFG
from src.models import (
    SHIPPED_FEATURES,
    Matcher,
    demonstrate_zero_probability_problem,
)


@pytest.fixture
def sample_pair_features():
    """Provides a synthetic 15-feature DataFrame representing 4 test pairs."""
    data = {
        "father_exact": [1.0, 0.0, 1.0, 0.0],
        "dob_exact": [1.0, 0.0, 0.0, 0.0],
        "dob_day_equal": [1.0, 0.0, 1.0, 0.0],
        "dob_month_equal": [1.0, 0.0, 1.0, 0.0],
        "dob_one_digit_edit": [0.0, 0.0, 1.0, 0.0],
        "dob_diff_days_log": [0.0, 5.8, 0.0, 7.2],
        "nid_hamming_le_1": [1.0, 0.0, 1.0, 0.0],
        "nid_levenshtein": [1.0, 0.2, 0.9, 0.1],
        "phone_last7_equal": [1.0, 0.0, 1.0, 0.0],
        "phone_edit_distance": [0.0, 8.0, 1.0, 9.0],
        "city_equal": [1.0, 0.0, 1.0, 0.0],
        "address_token_jaccard": [1.0, 0.1, 0.8, 0.0],
        "blood_group_equal": [1.0, 0.0, 1.0, 0.0],
        "visit_count_diff_log": [0.0, 2.5, 0.0, 4.1],
        "gower_similarity": [0.98, 0.15, 0.88, 0.08],
    }
    return pd.DataFrame(data)


@pytest.fixture
def superset_pair_features(sample_pair_features):
    """Provides a DataFrame with extra unselected columns (simulating full register comparison vector)."""
    df = sample_pair_features.copy()
    df["extra_feature_a"] = [99.0, 88.0, 77.0, 66.0]
    df["record_id_1"] = ["REC001", "REC002", "REC003", "REC004"]
    df["record_id_2"] = ["REC011", "REC012", "REC013", "REC014"]
    return df


def test_models_load_from_disk():
    """Asserts that all 4 model files exist in models/ and load into Matcher instances."""
    model_keys = ["decision_tree", "naive_bayes", "knn", "mlp"]

    for key in model_keys:
        model_path = CFG.MODELS_DIR / f"{key}.joblib"
        assert model_path.exists(), f"Missing model binary: {model_path}"

        matcher = Matcher.load(model_path)
        assert isinstance(matcher, Matcher), f"{key} is not a Matcher instance"
        assert matcher.feature_names == SHIPPED_FEATURES
        assert matcher.is_fitted, f"{key} should be marked fitted"

        # Check metadata sidecar
        meta_path = CFG.MODELS_DIR / f"{key}_meta.json"
        assert meta_path.exists(), f"Missing metadata file: {meta_path}"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert "chosen_params" in meta
        assert "cv_metrics" in meta
        assert "data_hash" in meta


def test_models_produce_valid_probabilities(sample_pair_features, superset_pair_features):
    """Asserts that all models return valid probabilities in [0, 1] summing to 1.0."""
    model_keys = ["decision_tree", "naive_bayes", "knn", "mlp"]

    for key in model_keys:
        matcher = Matcher.load(CFG.MODELS_DIR / f"{key}.joblib")

        # Test on exact 15 features
        probas = matcher.predict_proba(sample_pair_features)
        assert probas.shape == (len(sample_pair_features), 2), f"{key} probas shape mismatch"
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0), f"{key} probas out of [0, 1]"
        assert np.allclose(probas.sum(axis=1), 1.0), f"{key} probas do not sum to 1.0"

        # Pair 0 is an obvious duplicate (probas[:, 1] should be high)
        # Pair 3 is an obvious non-duplicate (probas[:, 1] should be low)
        assert probas[0, 1] > 0.70, f"{key} failed to recognize obvious duplicate"
        assert probas[3, 1] < 0.30, f"{key} failed to recognize obvious non-duplicate"

        # Test on superset DataFrame with extra columns
        probas_superset = matcher.predict_proba(superset_pair_features)
        assert np.allclose(probas, probas_superset), f"{key} extra columns affected predictions!"

        # Test predict with operating threshold
        preds_low = matcher.predict(sample_pair_features, threshold=0.20)
        preds_high = matcher.predict(sample_pair_features, threshold=0.80)
        assert set(preds_low.tolist()).issubset({0, 1})
        assert set(preds_high.tolist()).issubset({0, 1})


def test_models_deterministic(sample_pair_features):
    """Asserts that model inference is strictly deterministic on identical inputs."""
    model_keys = ["decision_tree", "naive_bayes", "knn", "mlp"]

    for key in model_keys:
        matcher = Matcher.load(CFG.MODELS_DIR / f"{key}.joblib")
        p1 = matcher.predict_proba(sample_pair_features)
        p2 = matcher.predict_proba(sample_pair_features)
        np.testing.assert_array_equal(p1, p2, err_msg=f"{key} is non-deterministic!")


def test_matcher_explain_stub(sample_pair_features):
    """Asserts that Matcher.explain produces structured clinical explanations."""
    matcher = Matcher.load(CFG.MODELS_DIR / "decision_tree.joblib")
    pair_row = sample_pair_features.iloc[0]

    explanation = matcher.explain(pair_row)
    assert isinstance(explanation, dict)
    assert "model_name" in explanation
    assert "match_probability" in explanation
    assert "verdict" in explanation
    assert "agreeing_features" in explanation
    assert "conflicting_features" in explanation
    assert "clerk_summary" in explanation

    # Verify no raw estimator internals leak into clerk explanation
    summary = explanation["clerk_summary"].lower()
    assert "node" not in summary
    assert "gini" not in summary
    assert "entropy" not in summary


def test_matcher_save_and_reload_roundtrip(sample_pair_features, tmp_path):
    """Asserts that Matcher can be serialized and reloaded without loss of functionality."""
    orig = Matcher.load(CFG.MODELS_DIR / "decision_tree.joblib")
    tmp_file = tmp_path / "test_dt.joblib"
    orig.save(tmp_file)

    reloaded = Matcher.load(tmp_file)
    p_orig = orig.predict_proba(sample_pair_features)
    p_reloaded = reloaded.predict_proba(sample_pair_features)
    np.testing.assert_array_equal(p_orig, p_reloaded)


def test_zero_probability_demonstration_effect():
    """Asserts that demonstrate_zero_probability_problem shows smoothing eliminates 0.0 collapse."""
    # Synthetic dataset with an asymmetric feature category
    rng = np.random.RandomState(42)
    X = rng.rand(100, len(SHIPPED_FEATURES))
    y = np.array([0] * 70 + [1] * 30)
    X[y == 1, 0] = 0.95
    X[y == 0, 0] = 0.10

    demo = demonstrate_zero_probability_problem(X, y, SHIPPED_FEATURES)
    assert "feature_tested" in demo
    assert "results" in demo
    results = demo["results"]

    unsmoothed = [r for r in results if r["alpha"] < 1e-4][0]
    smoothed = [r for r in results if r["alpha"] == 1.0][0]

    assert unsmoothed["collapsed_to_zero"], "Unsmoothed probability should collapse to zero"
    assert not smoothed["collapsed_to_zero"], "Laplace smoothed probability must not collapse"
    assert smoothed["p_duplicate"] > 0.0, "Laplace smoothed probability must be non-zero"


def test_model_selection_artifacts_exist_and_valid():
    """Asserts that reports/models.json, docs/model_selection.md, and metrics.json are populated."""
    # Check reports/models.json
    json_path = CFG.REPORTS_DIR / "models.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "comparison_table" in data
    assert len(data["comparison_table"]) == 4
    assert data["selection_objective"] == "PR-AUC (Average Precision)"
    assert data["one_standard_error_rule_applied"] is True
    assert "data_hash" in data

    # Check that all models have PR-AUC > 0.99
    for row in data["comparison_table"]:
        assert row["pr_auc_mean"] > 0.99, f"{row['family']} PR-AUC too low: {row['pr_auc_mean']}"

    # Check docs/model_selection.md
    doc_path = CFG.DOCS_DIR / "model_selection.md"
    assert doc_path.exists(), f"Missing {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "StratifiedGroupKFold" in content
    assert "One-Standard-Error" in content
    assert "Zero-Probability Problem" in content

    # Check reports/metrics.json has "models" key
    metrics_path = CFG.REPORTS_DIR / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "models" in metrics
    assert "tree_pr_auc" in metrics["models"]
    assert "nb_pr_auc" in metrics["models"]
    assert "knn_pr_auc" in metrics["models"]
    assert "mlp_pr_auc" in metrics["models"]
