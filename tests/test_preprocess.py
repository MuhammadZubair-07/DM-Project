"""Tests for preprocessing decisions, transformers, and zero-leakage pipeline builders.

Verifies:
1. Transformers fit strictly on training data only (no data leakage).
2. Pipelines transform test sets without refitting or altering fitted states.
3. Supervised entropy discretization and missing-value binning.
4. Redundant collinearity filtering and feature selection.
5. Decision tree scale-invariance (Lecture 4).
6. Artifact and report generation (reports/preprocessing.json, docs/preprocessing_decisions.md, figures).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src import config as CFG
from src.preprocess import (
    CorrelationFilterSelector,
    SupervisedEntropyDiscretizer,
    get_knn_pipeline,
    get_mlp_pipeline,
    get_nb_pipeline,
    get_tree_pipeline,
)


@pytest.fixture
def synthetic_train_test_data():
    """Provides synthetic train/test splits with known differences in distribution."""
    rng = np.random.RandomState(42)
    # Training set: 200 samples, 10 features, centered at 10.0
    X_train = rng.randn(200, 10) + 10.0
    y_train = (X_train[:, 0] + X_train[:, 1] > 20.0).astype(int)

    # Test set: 50 samples, centered at 50.0 (very different mean to test leakage)
    X_test = rng.randn(50, 10) + 50.0
    y_test = (X_test[:, 0] + X_test[:, 1] > 100.0).astype(int)

    return X_train, y_train, X_test, y_test


def test_transformers_fit_on_train_only(synthetic_train_test_data):
    """Asserts that transformers fit strictly on training data and test data never alters parameters."""
    X_train, y_train, X_test, y_test = synthetic_train_test_data

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_mean_before = np.copy(scaler.mean_)
    train_scale_before = np.copy(scaler.scale_)

    # Training mean should be close to 10.0, not 50.0
    assert np.allclose(train_mean_before, 10.0, atol=0.5)

    # Transform test set without refitting
    X_test_scaled = scaler.transform(X_test)

    # Assert fitted parameters did NOT change
    assert np.array_equal(scaler.mean_, train_mean_before), "Scaler mean was altered by transform(X_test)!"
    assert np.array_equal(scaler.scale_, train_scale_before), "Scaler scale was altered by transform(X_test)!"

    # Assert test set was scaled strictly using train mean and scale
    expected_test_scaled = (X_test - train_mean_before) / train_scale_before
    assert np.allclose(X_test_scaled, expected_test_scaled)


def test_pipeline_transforms_test_without_refitting(synthetic_train_test_data):
    """Asserts that all model family pipelines transform test sets without refitting."""
    X_train, y_train, X_test, y_test = synthetic_train_test_data

    pipelines = [
        ("tree", get_tree_pipeline(top_k=5)),
        ("knn", get_knn_pipeline(top_k=5)),
        ("nb", get_nb_pipeline(n_bins=5, min_categories=10)),
        ("mlp", get_mlp_pipeline(top_k=5)),
    ]

    for name, pipe in pipelines:
        # Fit strictly on train
        pipe.fit(X_train, y_train)

        # Inspect fitted parameters in transformers
        if "scaler" in pipe.named_steps:
            mean_before = np.copy(pipe.named_steps["scaler"].mean_)
        elif "discretizer" in pipe.named_steps:
            thresh_before = {k: list(v) for k, v in pipe.named_steps["discretizer"].thresholds_.items()}
        elif "selector" in pipe.named_steps:
            indices_before = list(pipe.named_steps["selector"].selected_indices_)

        # Predict on test
        preds = pipe.predict(X_test)
        probas = pipe.predict_proba(X_test)

        assert preds.shape == (len(X_test),), f"{name} predict shape mismatch"
        assert probas.shape == (len(X_test), 2), f"{name} predict_proba shape mismatch"
        assert np.allclose(probas.sum(axis=1), 1.0), f"{name} probabilities do not sum to 1.0"
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0), f"{name} probabilities out of range [0, 1]"

        # Verify transformer parameters were not modified during inference
        if "scaler" in pipe.named_steps:
            assert np.array_equal(pipe.named_steps["scaler"].mean_, mean_before), f"{name} scaler leaked test data!"
        elif "discretizer" in pipe.named_steps:
            assert pipe.named_steps["discretizer"].thresholds_ == thresh_before, f"{name} discretizer modified on test!"
        elif "selector" in pipe.named_steps:
            assert pipe.named_steps["selector"].selected_indices_ == indices_before, f"{name} selector modified on test!"


def test_supervised_entropy_discretizer_splits_and_nans():
    """Verifies MDLP discretizer splits on information gain and assigns explicit missing bin to NaNs."""
    rng = np.random.RandomState(42)
    # Feature 0: clear separation between class 0 and 1
    # Class 0: values near 0.2, Class 1: values near 0.8
    y = np.array([0] * 50 + [1] * 50)
    x0 = np.concatenate([rng.normal(0.2, 0.05, 50), rng.normal(0.8, 0.05, 50)])
    X = x0.reshape(-1, 1)

    discretizer = SupervisedEntropyDiscretizer(n_bins=3, min_samples_leaf=10)
    discretizer.fit(X, y)

    # Threshold should be placed near 0.5 where class entropy is minimized
    thresh = discretizer.thresholds_[0]
    assert len(thresh) > 0, "No entropy split found"
    assert any(0.3 <= t <= 0.7 for t in thresh), f"Expected threshold near 0.5, got {thresh}"

    # Test transform on unseen data including NaN
    X_test = np.array([[0.1], [0.5], [0.9], [np.nan]])
    binned = discretizer.transform(X_test)

    assert binned.shape == (4, 1)
    # NaN should be mapped to explicit missing bin (> valid threshold index)
    assert binned[3, 0] > 0, "NaN not mapped to dedicated missing bin"
    # Valid values should have monotonic bins
    assert binned[0, 0] <= binned[1, 0] <= binned[2, 0]


def test_correlation_filter_selector_drops_collinear():
    """Verifies that CorrelationFilterSelector drops redundant features with r > collinearity_threshold."""
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=100)

    # Feature 0: correlated with y
    f0 = y * 2.0 + rng.normal(0, 0.5, size=100)
    # Feature 1: almost exact duplicate of f0 (r > 0.99)
    f1 = f0 + rng.normal(0, 0.01, size=100)
    # Feature 2: independent random noise
    f2 = rng.randn(100)

    X = np.column_stack([f0, f1, f2])

    selector = CorrelationFilterSelector(collinearity_threshold=0.95, top_k=2)
    selector.fit(X, y)

    # One of f0 or f1 must be dropped due to collinearity
    selected = selector.selected_indices_
    assert len(selected) == 2
    assert not (0 in selected and 1 in selected), "Collinear features were not pruned!"

    # Test transform shape
    X_trans = selector.transform(X)
    assert X_trans.shape == (100, 2)


def test_correlation_filter_fixed_indices():
    """Verifies that fixed_indices overrides dynamic selection."""
    selector = CorrelationFilterSelector(fixed_indices=[0, 2])
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    selector.fit(X, y)
    assert selector.selected_indices_ == [0, 2]
    assert selector.transform(X).shape == (50, 2)


def test_tree_scale_invariance_lecture4():
    """Demonstrates Lecture 4 principle: Decision Trees are invariant to monotonic feature scaling."""
    rng = np.random.RandomState(42)
    X = rng.exponential(scale=10.0, size=(100, 4))
    y = (X[:, 0] > 12.0).astype(int)

    # Model 1 on unscaled data
    pipe_unscaled = get_tree_pipeline()
    pipe_unscaled.fit(X, y)
    preds_unscaled = pipe_unscaled.predict(X)

    # Model 2 on scaled data (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pipe_scaled = get_tree_pipeline()
    pipe_scaled.fit(X_scaled, y)
    preds_scaled = pipe_scaled.predict(X_scaled)

    # Decision tree predictions must be identical
    np.testing.assert_array_equal(preds_unscaled, preds_scaled)


def test_preprocessing_artifacts_exist():
    """Verifies that reports/preprocessing.json, figures, and docs/preprocessing_decisions.md exist and are valid."""
    json_path = CFG.REPORTS_DIR / "preprocessing.json"
    assert json_path.exists(), f"Missing {json_path}"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Check required experimental keys
    assert "rescaling" in data
    assert "pca_vs_selection" in data
    assert "discretization" in data
    assert "missing_values" in data
    assert "progressive_sampling" in data

    # Check specific values from Section 7.7 & 7.8
    assert data["rescaling"]["k-NN"]["pr_auc_gain"] > 0, "k-NN must benefit from scaling"
    assert abs(data["rescaling"]["Decision Tree"]["pr_auc_gain"]) < 1e-4, "Trees must be scale invariant"
    assert "Feature Selection" in data["pca_vs_selection"]["recommendation"]["chosen_strategy"]
    assert data["progressive_sampling"]["converged_sample_size"] == 3250

    # Check docs/preprocessing_decisions.md
    doc_path = CFG.DOCS_DIR / "preprocessing_decisions.md"
    assert doc_path.exists(), f"Missing {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "Feature Selection" in content
    assert "Progressive-Sampling Learning Curve" in content

    # Check figures
    expected_figs = [
        "rescaling_comparison.png",
        "pca_variance_explained.png",
        "pca_vs_selection.png",
        "discretization_comparison.png",
        "progressive_sampling_curve.png",
    ]
    for fig_name in expected_figs:
        fig_path = CFG.FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Empty figure: {fig_path}"
