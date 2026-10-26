"""Unit tests for the evaluation, calibration, and cost-threshold module (Lecture 4, Roadmap 7.9).

Verifies:
1. Metric calculations on hand-computed synthetic cases.
2. Bayesian and empirical cost threshold formulas:
       theta* = C_FP / (C_FP + C_FN)
3. Platt calibration monotonicity and output range [0, 1].
4. Expected Calibration Error (ECE) metric behavior.
5. Workload clerk-hour estimation.
6. Baseline alignment rule (majority baseline paired with accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as CFG
from src.evaluate import (
    PlattCalibrator,
    compute_cost_and_thresholds,
    compute_ece,
    evaluate_domain_rule_baseline,
)


# ── 1. Hand-Calculated Metric and Threshold Tests ──────────────────────────────

def test_cost_threshold_formula_hand_calculation() -> None:
    """Tests that the cost-derived optimal threshold formula matches hand derivation:
    
        theta* = C_FP / (C_FP + C_FN)
        
    For review band: C_FP=5.0, C_FN=20.0 => 5 / (5 + 20) = 5/25 = 0.20
    For strict band:  C_FP=50.0, C_FN=10.0 => 50 / (50 + 10) = 50/60 = 0.8333
    """
    y_true = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    p_pred = np.array([0.9, 0.8, 0.3, 0.15, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0])

    res = compute_cost_and_thresholds(
        y_true=y_true,
        p_calibrated=p_pred,
        c_fp=5.0,
        c_fn=20.0,
        c_fp_strict=50.0,
        c_fn_strict=10.0,
    )

    # Theoretical thresholds
    assert pytest.approx(res["cost_optimal_threshold_review"], 1e-4) == 0.20
    assert pytest.approx(res["cost_optimal_threshold_strict"], 1e-4) == 50.0 / 60.0

    # At threshold 0.20:
    # y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] (3 predicted positive)
    # TP = 1 (item 0)
    # FP = 2 (items 1, 2)
    # FN = 1 (item 3: y=1, p=0.15 < 0.20)
    # TN = 6 (items 4..9)
    # Total Cost = 2 * 5.0 + 1 * 20.0 = 10 + 20 = 30.0
    # Expected cost per 1000 pairs: (30.0 / 10) * 1000 = 3000.0
    cm = res["confusion_matrix_review"]
    assert cm["TP"] == 1
    assert cm["FP"] == 2
    assert cm["FN"] == 1
    assert cm["TN"] == 6

    assert pytest.approx(res["cost_per_1000_at_optimal"], 1e-2) == 3000.0
    assert pytest.approx(res["metrics_at_optimal"]["precision"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(res["metrics_at_optimal"]["recall"], 1e-4) == 0.50
    assert pytest.approx(res["metrics_at_optimal"]["accuracy"], 1e-4) == 0.70
    assert pytest.approx(res["metrics_at_optimal"]["majority_baseline"], 1e-4) == 0.80


def test_domain_rule_baseline_hand_calculation() -> None:
    """Verifies the simple domain rule baseline ('same DOB and same phone')."""
    records = [
        {"record_id_1": "R1", "record_id_2": "R2", "dob_diff_days": 0.0, "phone_exact_match": 1.0, "label": 1},
        {"record_id_1": "R3", "record_id_2": "R4", "dob_diff_days": 0.0, "phone_exact_match": 1.0, "label": 0},
        {"record_id_1": "R5", "record_id_2": "R6", "dob_diff_days": 5.0, "phone_exact_match": 1.0, "label": 1},
        {"record_id_1": "R7", "record_id_2": "R8", "dob_diff_days": 0.0, "phone_exact_match": 0.0, "label": 0},
    ]
    df_pairs = pd.DataFrame(records)
    res = evaluate_domain_rule_baseline(df_pairs)

    # 4 pairs:
    # Rule triggers when dob_diff_days == 0 and phone_exact_match == 1.0:
    # Pair 1: triggers (pred=1, label=1) -> TP
    # Pair 2: triggers (pred=1, label=0) -> FP
    # Pair 3: does not trigger (pred=0, label=1) -> FN
    # Pair 4: does not trigger (pred=0, label=0) -> TN
    # TP=1, FP=1, FN=1, TN=1
    assert res["TP"] == 1
    assert res["FP"] == 1
    assert res["FN"] == 1
    assert res["TN"] == 1
    assert pytest.approx(res["precision"], 1e-4) == 0.50
    assert pytest.approx(res["recall"], 1e-4) == 0.50
    assert pytest.approx(res["accuracy"], 1e-4) == 0.50
    assert pytest.approx(res["majority_baseline"], 1e-4) == 0.50


# ── 2. Calibration Metric & Calibrator Tests ──────────────────────────────────

def test_compute_ece_perfect_vs_uncalibrated() -> None:
    """Verifies that ECE is near zero for perfectly calibrated probabilities."""
    # Perfectly separated and calibrated
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p_perfect = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ece_perf = compute_ece(y_true, p_perfect, n_bins=5)
    assert pytest.approx(ece_perf, 1e-6) == 0.0

    # Heavily miscalibrated: overconfident wrong predictions
    p_bad = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    ece_bad = compute_ece(y_true, p_bad, n_bins=5)
    assert ece_bad > 0.5


def test_platt_calibrator_fitting_and_bounds() -> None:
    """Verifies PlattCalibrator transforms raw scores into strictly bounded [0, 1] probabilities."""
    cal = PlattCalibrator()
    raw_scores = np.array([0.05, 0.1, 0.2, 0.35, 0.7, 0.85, 0.9, 0.95])
    y_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    cal.fit(raw_scores, y_labels)
    calibrated = cal.transform(raw_scores)

    assert len(calibrated) == len(raw_scores)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    # Platt scaling is strictly monotonic: higher raw score => higher calibrated probability
    assert np.all(np.diff(calibrated) >= 0.0)


# ── 3. Workload and Assumption Validation ─────────────────────────────────────

def test_workload_hours_calculation() -> None:
    """Verifies the workload clerk-hours formula with configured assumptions."""
    n_pairs = 120
    minutes_per_review = CFG.MINUTES_PER_CLERK_REVIEW
    expected_hours = (n_pairs * minutes_per_review) / 60.0
    assert pytest.approx(expected_hours, 1e-4) == (120 * 3.0) / 60.0 == 6.0


# ── 4. End-to-End Evaluation Report Artifacts Existence ───────────────────────

def test_evaluation_artifacts_schema_and_integrity() -> None:
    """Verifies that reports/evaluation.json has been generated with all required sections."""
    eval_json_path = CFG.REPORTS_DIR / "evaluation.json"
    if not eval_json_path.exists():
        pytest.skip("evaluation.json not generated yet. Skipping until pipeline completes.")

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Core required sections
    required_keys = [
        "test_population",
        "baselines",
        "cross_validation_training",
        "test_models",
        "calibration_summary",
        "end_to_end",
        "workload_summary",
        "leakage_proof",
        "split_experiment",
        "robustness_check",
        "error_analysis",
        "generated_figures",
    ]
    for key in required_keys:
        assert key in data, f"Missing required evaluation key: {key}"

    # Verify baseline beside accuracy rule in test_models
    majority_base = data["baselines"]["majority_baseline"]["accuracy"]
    for model_name, metrics in data["test_models"].items():
        assert "majority_baseline" in metrics, f"Model {model_name} missing majority_baseline"
        assert pytest.approx(metrics["majority_baseline"], 1e-4) == majority_base

    # Check documentation file exists
    report_md = CFG.DOCS_DIR / "evaluation_report.md"
    assert report_md.exists(), "docs/evaluation_report.md must exist"
    assert report_md.stat().st_size > 1000, "docs/evaluation_report.md must not be empty"

    # Check figures exist
    fig_dir = CFG.REPORTS_DIR / "figures"
    expected_figs = [
        "roc_curves.png",
        "pr_curves.png",
        "confusion_matrices.png",
        "calibration_curves.png",
        "expected_cost_vs_threshold.png",
    ]
    for fig in expected_figs:
        assert (fig_dir / fig).exists(), f"Missing required figure: {fig}"
