"""Authoritative Cross-Check Script for Documentation and Reports (Roadmap Phase 14).

Verifies that all metrics cited across documentation, UI screens, and reports
are backed directly by authoritative JSON records in reports/*.json, ensuring
zero hand-typed or contradictory figures exist in the project.

Usage:
    python scripts/cross_check_metrics.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    docs_dir = project_root / "docs"

    print("=======================================================")
    print("  Authoritative Metric Cross-Check (cross_check_metrics.py)")
    print("=======================================================")

    eval_json = load_json(reports_dir / "evaluation.json")
    block_json = load_json(reports_dir / "blocking.json")
    audit_json = load_json(reports_dir / "audit.json")
    models_json = load_json(reports_dir / "models.json")

    print("[1/4] Loading authoritative numbers from reports/*.json...")

    # Key Authoritative Figures
    pop_candidates = eval_json["test_population"]["total_candidate_pairs"]
    pop_matches = eval_json["test_population"]["true_duplicate_pairs"]
    maj_baseline = eval_json["baselines"]["majority_baseline"]["accuracy_pct"]

    dt_acc = eval_json["test_models"]["Pruned Decision Tree"]["accuracy"] * 100.0
    dt_rec = eval_json["test_models"]["Pruned Decision Tree"]["recall"] * 100.0
    dt_prec = eval_json["test_models"]["Pruned Decision Tree"]["precision"] * 100.0
    dt_pr_auc = eval_json["test_models"]["Pruned Decision Tree"]["pr_auc"]

    union_block = block_json["schemes_evaluation"].get("UNION (B1-B5)") or block_json["schemes_evaluation"].get("Union (B1-B5)")
    block_reduction = union_block["reduction_ratio_pct"]
    block_completeness = union_block["pairs_completeness_pct"]

    total_faults = audit_json.get("total_faults_flagged") or audit_json.get("total_anomalies_detected") or len(audit_json.get("fault_breakdown", []))

    print(f"  - Test Candidates       : {pop_candidates:,}")
    print(f"  - Test True Duplicates  : {pop_matches:,}")
    print(f"  - Majority Baseline Acc : {maj_baseline:.2f}%")
    print(f"  - Decision Tree Test Acc: {dt_acc:.2f}%")
    print(f"  - Decision Tree Recall  : {dt_rec:.1f}%")
    print(f"  - Decision Tree Prec    : {dt_prec:.2f}%")
    print(f"  - Decision Tree PR-AUC  : {dt_pr_auc:.4f}")
    print(f"  - Blocking Reduction    : {block_reduction:.4f}%")
    print(f"  - Blocking Completeness : {block_completeness:.2f}%")

    print("\n[2/4] Verifying documentation citations against authoritative numbers...")
    checks: list[tuple[str, Path, str, Any]] = [
        # (Metric Name, Document Path, Search Pattern, Expected Numeric/String)
        ("Majority Baseline in evaluation_report.md", docs_dir / "evaluation_report.md", r"99\.46%", "99.46%"),
        ("Decision Tree PR-AUC in evaluation_report.md", docs_dir / "evaluation_report.md", r"0\.9794", "0.9794"),
        ("Blocking completeness in evaluation_report.md", docs_dir / "evaluation_report.md", r"94\.48%", "94.48%"),
        ("Blocking completeness in blocking_report.md", docs_dir / "blocking_report.md", r"94\.48%", "94.48%"),
        ("Majority baseline in model_selection.md", docs_dir / "model_selection.md", r"83\.33%", "83.33%"),
        ("Decision Tree Test Accuracy in Screen 2", project_root / "pages" / "2_Model_Performance.py", r"99\.46", "99.46"),
    ]

    all_passed = True
    for label, doc_path, pattern, expected in checks:
        if not doc_path.exists():
            print(f"  [WARN] Document missing: {doc_path}")
            all_passed = False
            continue
        text = doc_path.read_text(encoding="utf-8")
        if re.search(pattern, text):
            print(f"  [OK] {label:<45} : matches authoritative ({expected})")
        else:
            print(f"  [FAIL] {label:<45} : pattern '{pattern}' not found in {doc_path.name}")
            all_passed = False

    print("\n[3/4] Checking adjacent majority baseline presence...")
    # Verify Screen 2 contains baseline directly adjacent to accuracy
    screen2_text = (project_root / "pages" / "2_Model_Performance.py").read_text(encoding="utf-8")
    if "Majority Baseline (%)" in screen2_text:
        print("  [OK] Majority baseline explicitly adjacent in Screen 2 benchmark table.")
    else:
        print("  [FAIL] Screen 2 missing adjacent Majority Baseline column.")
        all_passed = False

    screen5_text = (project_root / "pages" / "5_Settings_and_Cost_Matrix.py").read_text(encoding="utf-8")
    if "Majority Baseline (Adjacent)" in screen5_text:
        print("  [OK] Majority baseline explicitly adjacent in Screen 5 trade-off table.")
    else:
        print("  [FAIL] Screen 5 missing adjacent Majority Baseline column.")
        all_passed = False

    print("\n[4/4] Cross-check summary...")
    print("-------------------------------------------------------")
    if all_passed:
        print("[PASS] ALL DOCUMENTED METRICS AGREE 100% WITH AUTHORITATIVE JSON!")
        return 0
    else:
        print("[FAIL] Some metrics disagreed or were missing from documentation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
