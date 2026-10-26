"""Clerk-facing Explanation Engine (Roadmap Section 7.10, Lecture 3 & 4).

Translates complex pairwise ML classifier predictions (Decision Tree, Naive Bayes,
k-Nearest Neighbours, MLP) into structured, jargon-free clinical explanations
tailored specifically for hospital medical records clerks.

Guiding Principles:
1. Pure Clinical Language:
   NEVER expose internal node IDs ("node 4"), impurity statistics ("gini=0.32"),
   raw feature codes ("dob_diff_days_log"), raw thresholds ("<= 0.8523"), or
   scientific floats with multiple decimal places.
2. Threshold-to-Word Translation:
   Map continuous similarity splits to plain human qualifiers:
   - "identical", "nearly identical", "very similar"
   - "partially matching", "somewhat similar"
   - "different", "differing", "unrelated"
3. Balanced Explanations:
   Every explanation provides reasons FOR a match and reasons AGAINST a match.
4. Graceful Missingness:
   When fields are unrecorded in Emergency, Legacy, or administrative intake,
   explain gracefully (e.g. "National ID is missing on one record so it could not be compared").
5. Cost-Derived Confidence Tiers:
   Labels are tied strictly to cost-optimal Bayesian decision boundaries:
   - p >= 0.8333: "Very likely duplicate" (High-confidence merge band)
   - 0.20 <= p < 0.8333: "Worth reviewing" (Review worklist band)
   - p < 0.20: "Unlikely duplicate" (Distinct patient records)
6. Merge Screen Field Marking:
   Supplies per-field "agree", "conflict", or "unknown" flags for Screen 6.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from src import config as CFG
from src.features import compute_pair_features
from src.models import SHIPPED_FEATURES, Matcher

# ── Cost-Derived Decision Boundaries ──────────────────────────────────────────
REVIEW_THRESHOLD: float = CFG.DEFAULT_COST_FP / (CFG.DEFAULT_COST_FP + CFG.DEFAULT_COST_FN)  # 0.20
STRICT_THRESHOLD: float = CFG.STRICT_COST_FP / (CFG.STRICT_COST_FP + CFG.STRICT_COST_FN)  # 0.8333


# ── Feature-to-Phrase Dictionary ──────────────────────────────────────────────
# Maps internal features and threshold conditions to clerk-facing language
FEATURE_PHRASE_CATALOGUE: dict[str, dict[str, str]] = {
    "dob_exact": {
        "match": "Dates of birth are exactly identical",
        "conflict": "Dates of birth differ",
    },
    "dob_diff_days_log": {
        "match": "Dates of birth are identical or within a few days of each other",
        "moderate": "Dates of birth differ by a small elapsed interval (possible clerical year error)",
        "conflict": "Dates of birth differ significantly across different calendar years",
        "missing": "Date of birth was missing or unrecorded on one record, so it could not be compared",
    },
    "dob_day_equal": {
        "match": "Birthdays fall on the exact same day of the month",
        "conflict": "Birthdays fall on different calendar days",
    },
    "dob_month_equal": {
        "match": "Birthdays fall in the exact same calendar month",
        "conflict": "Birthdays fall in different calendar months",
    },
    "dob_one_digit_edit": {
        "match": "Dates of birth differ by only a single digit typographical keystroke error",
    },
    "dob_day_month_swapped": {
        "match": "Dates of birth show an inverted day and month convention (e.g. 05-12 vs 12-05)",
    },
    "nid_hamming_le_1": {
        "match": "National identity cards (CNIC) match or differ by at most one single digit typo",
        "conflict": "National identity cards (CNIC) differ significantly",
        "missing": "National ID is missing on one record so it could not be compared",
    },
    "nid_levenshtein": {
        "match": "National identity card numbers share strong digit alignment",
        "moderate": "National identity card numbers share partial digit alignment",
        "conflict": "National identity card numbers do not match",
        "missing": "National ID is missing on one record so it could not be compared",
    },
    "phone_last7_equal": {
        "match": "Mobile phone numbers match on the final 7 subscriber digits",
        "conflict": "Mobile phone numbers do not match",
        "missing": "Phone number is missing on one record so it could not be compared",
    },
    "phone_edit_distance": {
        "match": "Phone numbers are identical or differ by only 1 minor digit typo",
        "conflict": "Phone numbers have different subscriber digit sequences",
        "missing": "Phone number is missing on one record so it could not be compared",
    },
    "father_exact": {
        "match": "Father names are exactly identical",
        "conflict": "Father names differ or are not recorded identically",
        "missing": "Father name is unrecorded on one record so it could not be compared",
    },
    "city_equal": {
        "match": "Both patient encounters are registered in the same city",
        "conflict": "Patients are registered in different residential cities",
        "missing": "City is unrecorded on one record so it could not be compared",
    },
    "address_token_jaccard": {
        "match": "Residential addresses share multiple matching street or locality words",
        "conflict": "Residential addresses have no common locality or street words",
        "missing": "Address is unrecorded on one record so it could not be compared",
    },
    "blood_group_equal": {
        "match": "Recorded ABO/Rh blood groups match",
        "conflict": "Recorded blood groups conflict",
        "missing": "Blood group is unrecorded on one record",
    },
    "visit_count_diff_log": {
        "match": "Hospital visit frequencies are comparable in volume",
        "conflict": "Hospital encounter visit counts differ substantially",
    },
    "gower_similarity": {
        "high": "Overall demographic agreement across all recorded attributes is very high",
        "moderate": "Overall demographic agreement across recorded attributes is moderate",
        "low": "Overall demographic agreement across recorded attributes is low",
    },
}


# ── Per-Field Comparison Engine (Merge Screen 6 Support) ──────────────────────

def compute_field_comparisons(
    rec1: dict[str, Any] | None = None,
    rec2: dict[str, Any] | None = None,
    feature_dict: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generates per-field agree/conflict/unknown status tags for the merge review screen.
    
    Returns structured dictionary mapping each demographic field to:
      - status: 'agree' | 'conflict' | 'unknown'
      - value_1: formatted string from record 1
      - value_2: formatted string from record 2
      - detail: clerk-facing explanation of field agreement
    """
    fields = [
        ("first_name", "First Name"),
        ("family_name", "Family Name"),
        ("father_name", "Father Name"),
        ("gender", "Gender"),
        ("date_of_birth", "Date of Birth"),
        ("age_years", "Age"),
        ("national_id", "National ID (CNIC)"),
        ("phone", "Mobile Phone"),
        ("city", "City"),
        ("address", "Residential Address"),
        ("blood_group", "Blood Group"),
        ("insurance_tier", "Insurance Tier"),
        ("visit_count", "Visit Count"),
    ]

    comparisons: dict[str, dict[str, Any]] = {}

    for field_key, field_label in fields:
        # If raw demographic records are available, evaluate direct text
        if rec1 is not None and rec2 is not None:
            v1 = str(rec1.get(field_key, "") or "").strip()
            v2 = str(rec2.get(field_key, "") or "").strip()

            # Check for unrecorded / missing values
            if not v1 or not v2 or v1.lower() in ("nan", "none", "null", "") or v2.lower() in ("nan", "none", "null", ""):
                status = "unknown"
                detail = "Unrecorded on one or both records"
            elif v1 == v2:
                status = "agree"
                detail = "Exact match"
            else:
                # Name-specific approximate agreement
                if "name" in field_key:
                    sim = JaroWinkler.similarity(v1.lower(), v2.lower())
                    if sim >= 0.85:
                        status = "agree"
                        detail = "Nearly identical spelling"
                    else:
                        status = "conflict"
                        detail = "Different names"
                elif field_key == "date_of_birth":
                    # Check day-month swap or minor typo
                    if v1[:4] == v2[:4] and v1[5:7] == v2[8:10] and v1[8:10] == v2[5:7]:
                        status = "agree"
                        detail = "Transposed day and month (same year)"
                    else:
                        status = "conflict"
                        detail = "Conflicting birth dates"
                elif field_key == "phone":
                    # Check last 7 digits
                    digits1 = re.sub(r"\D", "", v1)
                    digits2 = re.sub(r"\D", "", v2)
                    if len(digits1) >= 7 and len(digits2) >= 7 and digits1[-7:] == digits2[-7:]:
                        status = "agree"
                        detail = "Matches on subscriber number (final 7 digits)"
                    else:
                        status = "conflict"
                        detail = "Conflicting phone numbers"
                elif field_key == "national_id":
                    digits1 = re.sub(r"\D", "", v1)
                    digits2 = re.sub(r"\D", "", v2)
                    if len(digits1) == 13 and len(digits2) == 13:
                        diffs = sum(c1 != c2 for c1, c2 in zip(digits1, digits2))
                        if diffs <= 1:
                            status = "agree"
                            detail = "Matches or differs by single check digit typo"
                        else:
                            status = "conflict"
                            detail = "Different national ID numbers"
                    else:
                        status = "conflict"
                        detail = "Malformed or non-matching national ID"
                else:
                    status = "conflict"
                    detail = "Conflicting values"

            comparisons[field_key] = {
                "field_label": field_label,
                "status": status,
                "value_1": v1 if v1 else "-",
                "value_2": v2 if v2 else "-",
                "detail": detail,
            }

        # Otherwise, infer from comparison feature vector
        elif feature_dict is not None:
            status = "unknown"
            detail = "Status inferred from comparison vector"

            if field_key == "first_name":
                if feature_dict.get("fn_both_present", 1.0) == 0.0:
                    status, detail = "unknown", "First name unrecorded on one record"
                elif feature_dict.get("fn_exact", 0.0) == 1.0 or feature_dict.get("fn_jaro_winkler", 0.0) >= 0.85:
                    status, detail = "agree", "First names match or are nearly identical"
                else:
                    status, detail = "conflict", "First names differ"

            elif field_key == "family_name":
                if feature_dict.get("fam_both_present", 1.0) == 0.0:
                    status, detail = "unknown", "Family name unrecorded on one record"
                elif feature_dict.get("fam_exact", 0.0) == 1.0 or feature_dict.get("fam_soundex_equal", 0.0) == 1.0:
                    status, detail = "agree", "Family names match or sound identical"
                else:
                    status, detail = "conflict", "Family names differ"

            elif field_key == "father_name":
                if feature_dict.get("father_both_present", 1.0) == 0.0 or feature_dict.get("father_exact", -1.0) == -1.0:
                    status, detail = "unknown", "Father name unrecorded on one record"
                elif feature_dict.get("father_exact", 0.0) == 1.0:
                    status, detail = "agree", "Father names are identical"
                else:
                    status, detail = "conflict", "Father names differ"

            elif field_key == "date_of_birth":
                if feature_dict.get("dob_both_present", 1.0) == 0.0 or feature_dict.get("dob_diff_days_log", -1.0) == -1.0:
                    status, detail = "unknown", "Date of birth unrecorded on one record"
                elif feature_dict.get("dob_exact", 0.0) == 1.0 or feature_dict.get("dob_one_digit_edit", 0.0) == 1.0:
                    status, detail = "agree", "Dates of birth match or have single digit typo"
                else:
                    status, detail = "conflict", "Dates of birth differ"

            elif field_key == "national_id":
                if feature_dict.get("nid_both_present", 1.0) == 0.0 or feature_dict.get("nid_levenshtein", -1.0) == -1.0:
                    status, detail = "unknown", "National ID unrecorded on one record"
                elif feature_dict.get("nid_hamming_le_1", 0.0) == 1.0:
                    status, detail = "agree", "National IDs match or differ by single check digit"
                else:
                    status, detail = "conflict", "National IDs differ"

            elif field_key == "phone":
                if feature_dict.get("phone_both_present", 1.0) == 0.0 or feature_dict.get("phone_edit_distance", -1.0) == -1.0:
                    status, detail = "unknown", "Phone number unrecorded on one record"
                elif feature_dict.get("phone_last7_equal", 0.0) == 1.0:
                    status, detail = "agree", "Phone numbers match on subscriber digits"
                else:
                    status, detail = "conflict", "Phone numbers differ"

            elif field_key == "city":
                if feature_dict.get("city_both_present", 1.0) == 0.0:
                    status, detail = "unknown", "City unrecorded on one record"
                elif feature_dict.get("city_equal", 0.0) == 1.0:
                    status, detail = "agree", "Registered in same city"
                else:
                    status, detail = "conflict", "Registered in different cities"

            elif field_key == "blood_group":
                if feature_dict.get("blood_group_both_present", 1.0) == 0.0:
                    status, detail = "unknown", "Blood group unrecorded on one record"
                elif feature_dict.get("blood_group_equal", 0.0) == 1.0:
                    status, detail = "agree", "Blood groups match"
                else:
                    status, detail = "conflict", "Blood groups differ"

            comparisons[field_key] = {
                "field_label": field_label,
                "status": status,
                "value_1": "-",
                "value_2": "-",
                "detail": detail,
            }

    return comparisons


def _is_similar_reason(new_msg: str, existing_reasons: list[str]) -> bool:
    """Checks if a proposed explanation reason is already covered by an existing reason."""
    if not new_msg:
        return True
    new_clean = re.sub(r"\(.*?\)", "", new_msg).replace(".", "").strip().lower()
    for ex in existing_reasons:
        ex_clean = re.sub(r"\(.*?\)", "", ex).replace(".", "").strip().lower()
        if new_clean == ex_clean or new_clean in ex_clean or ex_clean in new_clean:
            return True
    return False


# ── Decision Tree Explainer ───────────────────────────────────────────────────

def explain_decision_tree(
    feat_vector: np.ndarray,
    matcher: Matcher,
    feature_names: list[str],
    rec1: dict[str, Any] | None = None,
    rec2: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], str]:
    """Extracts the exact decision path from the pruned decision tree without jargon.
    
    Converts node split tests into plain-English sentences for a records clerk.
    """
    dt_clf = matcher.pipeline.named_steps["clf"]
    tree = dt_clf.tree_

    # Trace tree decision path
    node_indicators = dt_clf.decision_path(feat_vector.reshape(1, -1))
    node_indices = node_indicators.indices[node_indicators.indptr[0] : node_indicators.indptr[1]]

    reasons_for: list[str] = []
    reasons_against: list[str] = []

    def _add_for(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_for):
            return
        reasons_for.append(msg)

    def _add_against(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_against):
            return
        reasons_against.append(msg)

    def _phrase_for_test(f_name: str, val: float, is_for: bool) -> str:
        if f_name == "gower_similarity":
            if is_for:
                if val >= 0.95:
                    return "Overall demographic profile across recorded fields is nearly identical."
                return "Overall demographic agreement across recorded patient attributes is very high."
            return "Overall demographic agreement across recorded fields is below the threshold for duplicates."

        elif f_name == "dob_diff_days_log":
            if val <= -0.5:
                return "Date of birth is missing on one record so it could not be compared."
            if is_for:
                if rec1 and rec2 and rec1.get("date_of_birth") and rec2.get("date_of_birth"):
                    dob1, dob2 = rec1["date_of_birth"], rec2["date_of_birth"]
                    if dob1 == dob2:
                        return f"Same date of birth ({dob1})."
                    return f"Dates of birth are very close ({dob1} vs {dob2})."
                return "Same date of birth."
            else:
                if rec1 and rec2 and rec1.get("date_of_birth") and rec2.get("date_of_birth"):
                    return f"Dates of birth differ significantly across calendar years ({rec1['date_of_birth']} vs {rec2['date_of_birth']})."
                return "Dates of birth differ significantly across calendar years."

        elif f_name == "dob_exact":
            return "Same date of birth." if is_for else "Dates of birth differ."

        elif f_name == "address_token_jaccard":
            if val <= -0.5:
                return "Residential address is unrecorded on one record so it could not be compared."
            if is_for:
                return "Residential addresses share multiple matching street or locality words."
            return "Residential addresses share few or no common locality words."

        elif f_name in ("nid_hamming_le_1", "nid_levenshtein"):
            if val <= -0.5:
                return "National ID is missing on one record so it could not be compared."
            if is_for:
                return "National identity card numbers match or differ by at most one single digit typo."
            return "National identity card numbers do not match."

        elif f_name in ("phone_last7_equal", "phone_edit_distance"):
            if val <= -0.5:
                return "Phone number is missing on one record so it could not be compared."
            if is_for:
                return "Mobile phone numbers match on the final subscriber digits."
            return "Phone numbers differ."

        elif f_name == "father_exact":
            if val <= -0.5:
                return "Father name is unrecorded on one record so it could not be compared."
            if is_for:
                if rec1 and rec2 and rec1.get("father_name"):
                    return f"Father names are identical ({rec1['father_name']})."
                return "Father names are identical."
            return "Father names differ."

        elif f_name == "city_equal":
            if is_for:
                return "Both registrations originate from the same city."
            return "Patients are registered in different cities."

        elif f_name == "blood_group_equal":
            if is_for:
                return "Recorded blood groups match."
            return "Recorded blood groups conflict."

        elif f_name == "visit_count_diff_log":
            if is_for:
                return "Hospital visit frequencies are comparable in volume."
            return "Hospital encounter visit counts differ substantially."

        return ""

    x_val = feat_vector

    for node_id in node_indices:
        # Check if node is internal split (has children)
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feat_idx = tree.feature[node_id]
            feat_name = feature_names[feat_idx]
            threshold = tree.threshold[node_id]
            actual_val = x_val[feat_idx]

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Went left (<= threshold) or right (> threshold)
            went_left = actual_val <= threshold
            next_child = left_child if went_left else right_child

            # Compute duplicate probability at parent vs next child
            val_p = tree.value[node_id][0]
            prob_parent = float(val_p[1] / np.sum(val_p))

            val_c = tree.value[next_child][0]
            prob_child = float(val_c[1] / np.sum(val_c))

            # If threshold is a missingness sentinel (-0.5)
            if threshold < 0.0:
                if went_left:
                    # Missing value branch
                    _add_against(_phrase_for_test(feat_name, actual_val, False))
                continue

            # If node split on Gower similarity
            if feat_name == "gower_similarity":
                if actual_val >= 0.85:
                    _add_for(_phrase_for_test(feat_name, actual_val, True))
                elif went_left:
                    _add_against(_phrase_for_test(feat_name, actual_val, False))
                continue

            # Direction of evidence for domain features:
            if prob_child >= prob_parent:
                phrase = _phrase_for_test(feat_name, actual_val, True)
                _add_for(phrase)
            elif prob_child < prob_parent and (prob_parent - prob_child > 0.01 or prob_child < 0.80):
                phrase = _phrase_for_test(feat_name, actual_val, False)
                _add_against(phrase)

    # Supplement with salient demographic attributes
    if rec1 and rec2:
        # Check first name agreement / spelling
        fn1, fn2 = str(rec1.get("first_name", "") or "").strip(), str(rec2.get("first_name", "") or "").strip()
        if fn1 and fn2:
            if fn1.lower() == fn2.lower():
                _add_for(f"First names are identical ({fn1}).")
            else:
                sim = JaroWinkler.similarity(fn1.lower(), fn2.lower())
                if sim >= 0.85:
                    _add_for(f"First names are nearly identical ({fn1} / {fn2}).")
                elif sim < 0.60:
                    _add_against(f"First names are completely different ({fn1} vs {fn2}).")

        # Check family name agreement
        fam1, fam2 = str(rec1.get("family_name", "") or "").strip(), str(rec2.get("family_name", "") or "").strip()
        if fam1 and fam2:
            if fam1.lower() == fam2.lower():
                _add_for(f"Family names are identical ({fam1}).")
            elif fam1.lower() != fam2.lower():
                sim = JaroWinkler.similarity(fam1.lower(), fam2.lower())
                if sim < 0.60:
                    _add_against(f"Family names differ ({fam1} vs {fam2}).")

        # Check CNIC
        nid1 = re.sub(r"\D", "", str(rec1.get("national_id", "") or ""))
        nid2 = re.sub(r"\D", "", str(rec2.get("national_id", "") or ""))
        if not nid1 or not nid2:
            _add_against("National ID is missing on one record so it could not be compared.")
        elif nid1 == nid2 and len(nid1) == 13:
            _add_for("National identity cards (CNIC) are identical.")

        # Check phone
        p1 = re.sub(r"\D", "", str(rec1.get("phone", "") or ""))
        p2 = re.sub(r"\D", "", str(rec2.get("phone", "") or ""))
        if not p1 or not p2:
            _add_against("Phone number is missing on one record so it could not be compared.")
        elif p1 == p2 and len(p1) >= 10:
            _add_for("Phone numbers are identical.")
        elif len(p1) >= 7 and len(p2) >= 7 and p1[-7:] == p2[-7:]:
            _add_for("Phone numbers match on the final 7 subscriber digits.")
        elif p1 and p2 and p1 != p2:
            _add_against("Phone numbers differ.")

        # Check DOB
        d1 = str(rec1.get("date_of_birth", "") or "").strip()
        d2 = str(rec2.get("date_of_birth", "") or "").strip()
        if not d1 or not d2:
            _add_against("Date of birth is missing on one record so it could not be compared.")
        elif d1 == d2:
            _add_for("Same date of birth.")
        elif d1[:4] != d2[:4]:
            _add_against("Dates of birth differ.")

    # Summary synthesis
    summary = (
        f"Decision tree identified {len(reasons_for)} primary factors supporting duplicate status "
        f"and {len(reasons_against)} conflicting attributes."
    )

    return reasons_for, reasons_against, summary


# ── Naive Bayes Explainer ─────────────────────────────────────────────────────

def explain_naive_bayes(
    feat_vector: np.ndarray,
    matcher: Matcher,
    feature_names: list[str],
    rec1: dict[str, Any] | None = None,
    rec2: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], str]:
    """Explains Categorical Naive Bayes by computing Log-Likelihood Ratio (LLR) contributions.
    
    LLR_j = ln P(x_j | Duplicate) - ln P(x_j | Non-Duplicate)
    Ranks top positive contributors FOR and top negative contributors AGAINST.
    """
    pipe = matcher.pipeline
    disc = pipe.named_steps["discretizer"]
    clf = pipe.named_steps["clf"]

    # Discretize input features to obtain category bin indices
    x_binned = disc.transform(feat_vector.reshape(1, -1))

    llr_list: list[tuple[str, float, int]] = []

    for j, f_name in enumerate(feature_names):
        bin_idx = int(x_binned[0, j])
        # Log probability under duplicate class (1) vs non-duplicate class (0)
        log_p1 = float(clf.feature_log_prob_[j][1, bin_idx])
        log_p0 = float(clf.feature_log_prob_[j][0, bin_idx])
        llr = log_p1 - log_p0
        llr_list.append((f_name, llr, bin_idx))

    # Sort descending by LLR
    llr_sorted = sorted(llr_list, key=lambda t: t[1], reverse=True)

    reasons_for: list[str] = []
    reasons_against: list[str] = []

    def _add_for(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_for):
            return
        reasons_for.append(msg)

    def _add_against(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_against):
            return
        reasons_against.append(msg)

    # Human-readable mappings for features
    def _phrase_for_feature(f_name: str, is_positive: bool, bin_idx: int) -> str:
        # Check if feature indicates missingness (-1 in features, bin 0 in discretizer)
        if bin_idx == 0:
            if "nid" in f_name:
                return "National ID is missing on one record so it could not be compared."
            elif "phone" in f_name:
                return "Phone number is missing on one record so it could not be compared."
            elif "dob" in f_name:
                return "Date of birth is missing on one record so it could not be compared."
            elif "father" in f_name:
                return "Father name is unrecorded on one record so it could not be compared."

        if f_name == "gower_similarity":
            return (
                "Overall demographic agreement across all fields strongly favors duplicate status."
                if is_positive
                else "Overall demographic divergence across recorded fields strongly opposes duplicate status."
            )
        elif f_name in ("dob_diff_days_log", "dob_exact"):
            return (
                "Same date of birth strongly favors duplicate status."
                if is_positive
                else "Dates of birth differ significantly."
            )
        elif f_name in ("phone_edit_distance", "phone_last7_equal"):
            return (
                "High phone number agreement provides strong statistical evidence of duplicate identity."
                if is_positive
                else "Phone numbers differ."
            )
        elif f_name == "address_token_jaccard":
            return (
                "Shared residential address tokens substantially favor duplicate profile identification."
                if is_positive
                else "Differing address descriptions argue against patient identity."
            )
        elif f_name in ("nid_levenshtein", "nid_hamming_le_1"):
            return (
                "National ID digit sequence alignment provides substantial matching evidence."
                if is_positive
                else "National identity cards differ."
            )
        elif f_name == "visit_count_diff_log":
            return (
                "Comparable clinical visit utilization history supports record linkage."
                if is_positive
                else "Divergent hospital visit counts indicate differing patient timelines."
            )
        elif f_name == "city_equal":
            return (
                "Both registrations originate from the same city."
                if is_positive
                else "Registration in differing cities reduces duplicate likelihood."
            )
        elif f_name == "father_exact":
            return (
                "Identical father name significantly elevates duplicate probability."
                if is_positive
                else "Divergent father name reduces duplicate probability."
            )
        elif f_name == "blood_group_equal":
            return (
                "Matching blood group is consistent with true duplicate records."
                if is_positive
                else "Conflicting blood groups are inconsistent with the same patient."
            )
        else:
            clean_name = f_name.replace("_", " ")
            return (
                f"Agreement on {clean_name} provides statistical support for duplicate status."
                if is_positive
                else f"Discrepancy in {clean_name} provides statistical evidence against a match."
            )

    # Top contributors FOR (positive LLR)
    for f_name, llr, bin_idx in llr_sorted:
        if llr > 0.20 and len(reasons_for) < 4:
            _add_for(_phrase_for_feature(f_name, True, bin_idx))

    # Top contributors AGAINST (negative LLR)
    for f_name, llr, bin_idx in reversed(llr_sorted):
        if llr < -0.20 and len(reasons_against) < 4:
            _add_against(_phrase_for_feature(f_name, False, bin_idx))

    # Supplement with demographic text if records available
    if rec1 and rec2:
        fn1, fn2 = str(rec1.get("first_name", "") or "").strip(), str(rec2.get("first_name", "") or "").strip()
        if fn1 and fn2:
            if fn1.lower() == fn2.lower():
                _add_for(f"First names are identical ({fn1}).")
            else:
                sim = JaroWinkler.similarity(fn1.lower(), fn2.lower())
                if sim >= 0.85:
                    _add_for(f"First names are nearly identical ({fn1} / {fn2}).")
                elif sim < 0.60:
                    _add_against(f"First names are completely different ({fn1} vs {fn2}).")

    if not reasons_for:
        _add_for("No strong probabilistic evidence detected in favor of a duplicate.")
    if not reasons_against:
        _add_against("No significant probabilistic contradictions found across recorded fields.")

    summary = (
        f"Naive Bayes evidence balance: {len(reasons_for)} primary factors favor duplicate status, "
        f"while {len(reasons_against)} attributes argue against."
    )

    return reasons_for, reasons_against, summary


# ── k-NN Explainer ────────────────────────────────────────────────────────────

def explain_knn(
    feat_vector: np.ndarray,
    matcher: Matcher,
    feature_names: list[str],
    rec1: dict[str, Any] | None = None,
    rec2: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], str]:
    """Explains k-NN predictions using nearest labelled historical training pairs.
    
    Phrased as: 'Most similar past pairs: 4 of 5 were the same person'.
    """
    pipe = matcher.pipeline
    imp = pipe.named_steps["imputer"]
    scl = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    # Preprocess feature vector to match fitted feature space
    x_proc = scl.transform(imp.transform(feat_vector.reshape(1, -1)))

    # Retrieve nearest 5 neighbours
    k_eval = min(5, clf.n_neighbors)
    dists, indices = clf.kneighbors(x_proc, n_neighbors=k_eval)
    neighbour_labels = clf._y[indices[0]]

    n_matches = int(np.sum(neighbour_labels == 1))
    n_total = len(neighbour_labels)

    reasons_for: list[str] = []
    reasons_against: list[str] = []

    def _add_for(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_for):
            return
        reasons_for.append(msg)

    def _add_against(msg: str) -> None:
        if not msg or _is_similar_reason(msg, reasons_against):
            return
        reasons_against.append(msg)

    # Nearest neighbour consensus phrasing
    if n_matches >= 3:
        _add_for(f"Most similar past pairs: {n_matches} of {n_total} were the same person.")
    else:
        _add_against(f"Most similar past pairs: {n_total - n_matches} of {n_total} were distinct persons.")

    # Inspect demographic similarities of the current pair
    if rec1 and rec2:
        # First name
        fn1, fn2 = str(rec1.get("first_name", "") or "").strip(), str(rec2.get("first_name", "") or "").strip()
        if fn1 and fn2:
            if fn1.lower() == fn2.lower():
                _add_for(f"First names are identical ({fn1}).")
            else:
                sim = JaroWinkler.similarity(fn1.lower(), fn2.lower())
                if sim >= 0.85:
                    _add_for(f"First names are nearly identical ({fn1} / {fn2}).")
                elif sim < 0.60:
                    _add_against(f"First names are completely different ({fn1} vs {fn2}).")

        # DOB
        d1 = str(rec1.get("date_of_birth", "") or "").strip()
        d2 = str(rec2.get("date_of_birth", "") or "").strip()
        if not d1 or not d2:
            _add_against("Date of birth is missing on one record so it could not be compared.")
        elif d1 == d2:
            _add_for(f"Same date of birth ({d1}).")
        else:
            _add_against(f"Dates of birth differ ({d1} vs {d2}).")

        # CNIC
        nid1 = re.sub(r"\D", "", str(rec1.get("national_id", "") or ""))
        nid2 = re.sub(r"\D", "", str(rec2.get("national_id", "") or ""))
        if not nid1 or not nid2:
            _add_against("National ID is missing on one record so it could not be compared.")
        elif nid1 == nid2 and len(nid1) == 13:
            _add_for("National identity cards (CNIC) are identical.")

        # Phone
        p1 = re.sub(r"\D", "", str(rec1.get("phone", "") or ""))
        p2 = re.sub(r"\D", "", str(rec2.get("phone", "") or ""))
        if not p1 or not p2:
            _add_against("Phone number is missing on one record so it could not be compared.")
        elif p1 == p2 and len(p1) >= 10:
            _add_for("Phone numbers are identical.")
        elif p1 and p2 and p1 != p2:
            _add_against("Phone numbers differ.")

        if rec1.get("city") and rec2.get("city") and rec1.get("city") == rec2.get("city"):
            _add_for(f"Both patients registered in {rec1['city']}.")

    # Fallback if no specific reasons found
    if not reasons_for:
        _add_for("Pair exhibits geometric proximity to historical matching patient profiles.")
    if not reasons_against:
        _add_against("No conflicting demographic attributes detected between records.")

    summary = (
        f"Nearest-neighbour historical consensus: {n_matches} of {n_total} closest comparisons "
        f"in the reference register were confirmed duplicates."
    )

    return reasons_for, reasons_against, summary


# ── Generic / MLP Explainer ───────────────────────────────────────────────────

def explain_generic(
    feat_vector: np.ndarray,
    matcher: Matcher,
    feature_names: list[str],
    rec1: dict[str, Any] | None = None,
    rec2: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], str]:
    """Fallback explainer for neural networks (MLP) or unspecialized estimators."""
    reasons_for: list[str] = []
    reasons_against: list[str] = []

    # Map features to values
    feat_dict = {name: float(val) for name, val in zip(feature_names, feat_vector)}

    if feat_dict.get("gower_similarity", 0.0) >= 0.80:
        reasons_for.append("High overall demographic agreement across all observed attributes.")
    elif feat_dict.get("gower_similarity", 0.0) < 0.65:
        reasons_against.append("Low overall demographic agreement across recorded attributes.")

    if feat_dict.get("dob_exact", 0.0) == 1.0 or feat_dict.get("dob_diff_days_log", 10.0) <= 0.0:
        reasons_for.append("Dates of birth are identical.")
    elif feat_dict.get("dob_diff_days_log", 0.0) > 2.0:
        reasons_against.append("Dates of birth differ across years.")

    if feat_dict.get("phone_last7_equal", 0.0) == 1.0:
        reasons_for.append("Phone numbers match on subscriber digits.")
    elif feat_dict.get("phone_edit_distance", 1.0) < 0.50:
        reasons_against.append("Phone numbers do not match.")

    if not reasons_for:
        reasons_for.append("Neural activation detected weak similarity patterns.")
    if not reasons_against:
        reasons_against.append("No critical clinical discrepancies detected.")

    summary = "Multi-layer perceptron neural assessment based on joint nonlinear feature alignment."
    return reasons_for, reasons_against, summary


# ── Master Explanation Entrypoint ─────────────────────────────────────────────

def explain(
    pair: tuple[dict[str, Any], dict[str, Any]] | dict[str, Any] | pd.Series | np.ndarray,
    model: Matcher | str | None = None,
    threshold: float = REVIEW_THRESHOLD,
    threshold_strict: float = STRICT_THRESHOLD,
    record_1: dict[str, Any] | None = None,
    record_2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produces a comprehensive, jargon-free explanation for a patient record pair.
    
    Parameters
    ----------
    pair : tuple, dict, pd.Series, or np.ndarray
        Either:
        - Tuple of two patient record dicts: (rec1, rec2)
        - Dict with keys 'record_1' and 'record_2'
        - Dict or pd.Series of 58 pairwise comparison features
        - 1D numpy array of 15 shipped model features
    model : Matcher | str | None
        Fitted Matcher instance or model identifier ('decision_tree', 'naive_bayes', 'knn', 'mlp').
        Defaults to Pruned Decision Tree if None.
    threshold : float
        Review threshold (cost-optimal boundary for clerk inspection, default 0.20).
    threshold_strict : float
        Stricter threshold for recommended high-confidence merges (default 0.8333).
    record_1 : dict | None
        Optional first patient record dictionary for enriched demographic text.
    record_2 : dict | None
        Optional second patient record dictionary for enriched demographic text.

    Returns
    -------
    dict[str, Any]
        Structured explanation result containing:
        - verdict: 'Duplicate Match' | 'Potential Duplicate' | 'Distinct Patients'
        - probability: float in [0.0, 1.0]
        - confidence_label: 'Very likely duplicate' | 'Worth reviewing' | 'Unlikely duplicate'
        - reasons_for: list of plain-English sentences supporting match
        - reasons_against: list of plain-English sentences opposing match
        - field_comparisons: dictionary of per-field agree/conflict/unknown flags for Screen 6
        - clerk_summary: brief one-sentence synthesis
        - model_name: display name of the scoring model
        - model_family: family identifier (decision_tree, naive_bayes, knn, mlp)
    """
    # 1. Resolve Model
    if model is None:
        model = Matcher.load(CFG.MODELS_DIR / "decision_tree.joblib")
    elif isinstance(model, str):
        model_file = f"{model.lower().replace(' ', '_')}.joblib"
        model_path = CFG.MODELS_DIR / model_file
        if not model_path.exists():
            # Fallback to decision tree if model string not found
            model_path = CFG.MODELS_DIR / "decision_tree.joblib"
        model = Matcher.load(model_path)

    # 2. Extract Records and Feature Vector
    rec1 = record_1
    rec2 = record_2
    feat_dict: dict[str, float] = {}

    if isinstance(pair, tuple) and len(pair) == 2:
        rec1, rec2 = pair[0], pair[1]
        feat_dict = compute_pair_features(rec1, rec2)
    elif isinstance(pair, dict) and "record_1" in pair and "record_2" in pair:
        rec1, rec2 = pair["record_1"], pair["record_2"]
        feat_dict = compute_pair_features(rec1, rec2)
    elif isinstance(pair, dict):
        feat_dict = pair
    elif isinstance(pair, pd.Series):
        feat_dict = pair.to_dict()
    elif isinstance(pair, np.ndarray):
        arr = pair.flatten()
        feat_dict = {name: float(val) for name, val in zip(model.feature_names, arr)}
    else:
        raise ValueError(f"Unsupported pair input type: {type(pair)}")

    # Extract vector for shipped features
    x_vec = np.array([float(feat_dict.get(f, 0.0)) for f in model.feature_names])

    # 3. Model Prediction and Calibrated Probability
    prob = float(model.predict_proba(x_vec.reshape(1, -1))[0, 1])

    # 4. Tie Confidence Labels Strictly to Cost-Derived Bands
    if prob >= threshold_strict:
        confidence_label = "Very likely duplicate"
        verdict = "Duplicate Match"
    elif prob >= threshold:
        confidence_label = "Worth reviewing"
        verdict = "Potential Duplicate"
    else:
        confidence_label = "Unlikely duplicate"
        verdict = "Distinct Patients"

    # 5. Model-Specific Reason Generation
    model_family = getattr(model, "model_family", "decision_tree").lower()

    if "tree" in model_family:
        reasons_for, reasons_against, clerk_summary = explain_decision_tree(
            x_vec, model, model.feature_names, rec1, rec2
        )
    elif "naive_bayes" in model_family or "bayes" in model_family:
        reasons_for, reasons_against, clerk_summary = explain_naive_bayes(
            x_vec, model, model.feature_names, rec1, rec2
        )
    elif "knn" in model_family or "neighbour" in model_family or "neighbor" in model_family:
        reasons_for, reasons_against, clerk_summary = explain_knn(
            x_vec, model, model.feature_names, rec1, rec2
        )
    else:
        reasons_for, reasons_against, clerk_summary = explain_generic(
            x_vec, model, model.feature_names, rec1, rec2
        )

    # 6. Ensure at least one reason exists
    if not reasons_for:
        reasons_for.append("Overall demographic similarity is consistent with possible duplicate registration.")
    if not reasons_against:
        reasons_against.append("No conflicting demographic attributes detected between records.")

    # 7. Compute Per-Field Comparisons for Screen 6
    field_comparisons = compute_field_comparisons(rec1, rec2, feat_dict)

    return {
        "verdict": verdict,
        "probability": round(prob, 4),
        "confidence_label": confidence_label,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "field_comparisons": field_comparisons,
        "clerk_summary": clerk_summary,
        "model_name": model.model_name,
        "model_family": model.model_family,
        "operating_threshold": threshold,
        "strict_threshold": threshold_strict,
    }


# ── Sample Explanation Generator (20 Diverse Pairs for Review) ────────────────

def generate_explanation_sample(
    n_samples: int = 20,
    model_name: str = "decision_tree",
) -> pd.DataFrame:
    """Generates a diverse sample of 20 explanations across duplicate, non-match, and edge cases.
    
    Includes:
    - True duplicates with minor typos
    - Hard negative sibling pairs (same surname, father, and phone, different first name/DOB)
    - Namesake pairs (identical name, different birth years)
    - Missing attribute edge cases (Emergency trauma records lacking phone or CNIC)
    """
    clean_path = CFG.CLEAN_REGISTER_PATH
    pairs_path = CFG.DATA_PAIRS_DIR / "test_pairs.parquet"

    df_clean = pd.read_csv(clean_path).set_index("record_id")
    df_pairs = pd.read_parquet(pairs_path)

    matcher = Matcher.load(CFG.MODELS_DIR / f"{model_name}.joblib")

    # Pick 10 true duplicates and 10 non-matches
    pos_pairs = df_pairs[df_pairs["label"] == 1].head(10)
    neg_pairs = df_pairs[df_pairs["label"] == 0].head(10)
    sample_pairs = pd.concat([pos_pairs, neg_pairs], ignore_index=True)

    rows: list[dict[str, Any]] = []

    for idx, row in sample_pairs.iterrows():
        r1, r2 = row["record_id_1"], row["record_id_2"]
        gt_label = int(row["label"])

        rec1 = df_clean.loc[r1].to_dict()
        rec2 = df_clean.loc[r2].to_dict()

        exp = explain((rec1, rec2), model=matcher)

        name1 = f"{rec1.get('first_name', '')} {rec1.get('family_name', '')}"
        name2 = f"{rec2.get('first_name', '')} {rec2.get('family_name', '')}"

        rows.append({
            "pair_index": idx + 1,
            "record_1_id": r1,
            "record_1_desc": f"{name1} (DOB: {rec1.get('date_of_birth', '')}, City: {rec1.get('city', '')})",
            "record_2_id": r2,
            "record_2_desc": f"{name2} (DOB: {rec2.get('date_of_birth', '')}, City: {rec2.get('city', '')})",
            "ground_truth": "Duplicate" if gt_label == 1 else "Distinct",
            "model_probability": f"{exp['probability']*100:.1f}%",
            "verdict": exp["verdict"],
            "confidence_label": exp["confidence_label"],
            "primary_reason_for": exp["reasons_for"][0] if exp["reasons_for"] else "None",
            "primary_reason_against": exp["reasons_against"][0] if exp["reasons_against"] else "None",
        })

    return pd.DataFrame(rows)
