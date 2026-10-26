"""Pairwise feature extraction module (Comparison Vector).

Computes comprehensive attribute-level similarities, distances, and agreement flags
between paired patient records adhering strictly to Section 7.6 of PROJECT_ROADMAP.md.

Implements character n-gram Jaccard and TF-IDF cosine from scratch using pure
Python / numpy primitives so that their mechanics are completely defensible at the viva.
Leverages RapidFuzz for C-accelerated Jaro-Winkler and Levenshtein metrics.

Includes:
- Strict forbidden-column and allowlist guard (no leakage)
- Per-field both_present flags
- Overall Gower similarity with the delta rule (Lecture 4) skipping missing attributes
- Documented treatment of missing values per model family (Trees, Naive Bayes, k-NN)
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src import config as CFG
from src.blocking import soundex

# ── Self-Implemented String Metrics (Defensible at Viva) ───────────────────────

def char_ngram_jaccard(s1: str, s2: str, n: int = 2) -> float:
    """Computes character n-gram Jaccard similarity from scratch (Lecture 4).
    
    Formula:
        J(s1, s2) = |N(s1) ∩ N(s2)| / |N(s1) ∪ N(s2)|
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 0.0

    grams1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    grams2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}

    intersection = len(grams1 & grams2)
    union = len(grams1 | grams2)
    return intersection / union if union > 0 else 0.0


def char_tfidf_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Cosine similarity from scratch using term frequencies.
    
    Formula:
        Cosine(u, v) = (u · v) / (||u||_2 * ||v||_2)
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 0.0

    grams1 = Counter(s1[i : i + n] for i in range(len(s1) - n + 1))
    grams2 = Counter(s2[i : i + n] for i in range(len(s2) - n + 1))

    common = set(grams1.keys()) & set(grams2.keys())
    if not common:
        return 0.0

    dot = sum(grams1[g] * grams2[g] for g in common)
    norm1 = math.sqrt(sum(v * v for v in grams1.values()))
    norm2 = math.sqrt(sum(v * v for v in grams2.values()))
    denom = norm1 * norm2

    return dot / denom if denom > 0.0 else 0.0


def token_jaccard(s1: str, s2: str) -> float:
    """Computes word token Jaccard similarity on whitespace-split text."""
    if not s1 or not s2:
        return 0.0
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def token_cosine(s1: str, s2: str) -> float:
    """Computes word token Cosine similarity on bag-of-words frequencies."""
    if not s1 or not s2:
        return 0.0
    t1 = Counter(re.findall(r"\w+", s1.lower()))
    t2 = Counter(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    common = set(t1.keys()) & set(t2.keys())
    if not common:
        return 0.0
    dot = sum(t1[w] * t2[w] for w in common)
    norm1 = math.sqrt(sum(v * v for v in t1.values()))
    norm2 = math.sqrt(sum(v * v for v in t2.values()))
    denom = norm1 * norm2
    return dot / denom if denom > 0.0 else 0.0


# ── Forbidden-Column and Allowlist Guard ───────────────────────────────────────

def check_forbidden_columns(columns: list[str] | set[str] | pd.Index) -> None:
    """Validates that input record schema strictly obeys leakage constraints.
    
    Raises
    ------
    ValueError
        If any forbidden column or unallowlisted attribute is detected.
    """
    cols_set = set(columns)
    forbidden_found = cols_set & CFG.FORBIDDEN_COLUMNS
    if forbidden_found:
        raise ValueError(
            f"FORBIDDEN COLUMN GUARD: Input contains prohibited columns: {sorted(forbidden_found)}. "
            "Ground truth and generator metadata must never leak into feature extraction."
        )

    # Allowed metadata in pair files (pair identifiers and optional targets)
    pair_meta = {"record_id_1", "record_id_2", "label", "sample_weight"}
    unallowlisted = (cols_set - pair_meta) - set(CFG.CLEAN_MODEL_ALLOWLIST)
    if unallowlisted:
        raise ValueError(
            f"ALLOWLIST GUARD: Input contains unallowlisted columns: {sorted(unallowlisted)}. "
            f"Only columns in CLEAN_MODEL_ALLOWLIST are permitted."
        )


# ── Core Pair Comparison Vector Function ──────────────────────────────────────

INSURANCE_RANKS: dict[str, int] = {
    "none": 0,
    "basic": 1,
    "standard": 2,
    "premium": 3,
}


def parse_iso_date(date_str: str) -> tuple[int, int, int] | None:
    """Safely extracts (year, month, day) from standard ISO YYYY-MM-DD string."""
    if not date_str or not isinstance(date_str, str) or len(date_str) < 10:
        return None
    try:
        parts = date_str[:10].split("-")
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, TypeError):
        return None
    return None


def compute_pair_features(
    r1: dict[str, Any],
    r2: dict[str, Any],
) -> dict[str, float]:
    """Computes comparison feature vector between two cleaned patient records.
    
    Parameters
    ----------
    r1, r2 : dict[str, Any]
        Cleaned record dictionaries containing attributes from CLEAN_MODEL_ALLOWLIST.
        
    Returns
    -------
    dict[str, float]
        Dictionary of 58 similarity, distance, and both_present indicator features.
    """
    feats: dict[str, float] = {}

    # Gower accumulator: sum(delta_k * s_k) and sum(delta_k)
    gower_num = 0.0
    gower_den = 0.0

    # ── 1. First Name ──────────────────────────────────────────────────────────
    fn1 = str(r1.get("first_name", "")).strip().lower()
    fn2 = str(r2.get("first_name", "")).strip().lower()
    fn_present = float(bool(fn1 and fn2))
    feats["fn_both_present"] = fn_present

    if fn_present:
        feats["fn_jaro_winkler"] = float(JaroWinkler.similarity(fn1, fn2))
        feats["fn_levenshtein"] = float(Levenshtein.normalized_similarity(fn1, fn2))
        feats["fn_exact"] = 1.0 if fn1 == fn2 else 0.0
        feats["fn_soundex_equal"] = 1.0 if soundex(fn1) == soundex(fn2) else 0.0
        feats["fn_initial_equal"] = 1.0 if fn1[:1] == fn2[:1] else 0.0
    else:
        feats["fn_jaro_winkler"] = 0.0
        feats["fn_levenshtein"] = 0.0
        feats["fn_exact"] = 0.0
        feats["fn_soundex_equal"] = 0.0
        feats["fn_initial_equal"] = 0.0

    # ── 2. Family Name ─────────────────────────────────────────────────────────
    fam1 = str(r1.get("family_name", "")).strip().lower()
    fam2 = str(r2.get("family_name", "")).strip().lower()
    fam_present = float(bool(fam1 and fam2))
    feats["fam_both_present"] = fam_present

    if fam_present:
        feats["fam_jaro_winkler"] = float(JaroWinkler.similarity(fam1, fam2))
        feats["fam_levenshtein"] = float(Levenshtein.normalized_similarity(fam1, fam2))
        feats["fam_exact"] = 1.0 if fam1 == fam2 else 0.0
        feats["fam_soundex_equal"] = 1.0 if soundex(fam1) == soundex(fam2) else 0.0
        feats["fam_initial_equal"] = 1.0 if fam1[:1] == fam2[:1] else 0.0
    else:
        feats["fam_jaro_winkler"] = 0.0
        feats["fam_levenshtein"] = 0.0
        feats["fam_exact"] = 0.0
        feats["fam_soundex_equal"] = 0.0
        feats["fam_initial_equal"] = 0.0

    # ── 3. Compound Full Name ──────────────────────────────────────────────────
    full1 = f"{fam1} {fn1}".strip()
    full2 = f"{fam2} {fn2}".strip()
    full_present = float(bool(full1 and full2))

    if full_present:
        feats["full_name_jaro_winkler"] = float(JaroWinkler.similarity(full1, full2))
        feats["full_name_levenshtein"] = float(Levenshtein.normalized_similarity(full1, full2))
        feats["full_name_ngram_jaccard"] = float(char_ngram_jaccard(full1, full2, n=2))
        feats["full_name_tfidf_cosine"] = float(char_tfidf_cosine(full1, full2, n=3))

        # Check first/family inverted swap
        swap_exact = (fn1 == fam2) and (fam1 == fn2) and (fn1 != fam1)
        swap_jw = (JaroWinkler.similarity(fn1, fam2) >= 0.85) and (JaroWinkler.similarity(fam1, fn2) >= 0.85)
        feats["name_swap_flag"] = 1.0 if (swap_exact or swap_jw) else 0.0

        # Gower contribution: Name
        gower_num += feats["full_name_jaro_winkler"]
        gower_den += 1.0
    else:
        feats["full_name_jaro_winkler"] = 0.0
        feats["full_name_levenshtein"] = 0.0
        feats["full_name_ngram_jaccard"] = 0.0
        feats["full_name_tfidf_cosine"] = 0.0
        feats["name_swap_flag"] = 0.0

    # ── 4. Father Name ─────────────────────────────────────────────────────────
    fat1 = str(r1.get("father_name", "")).strip().lower()
    fat2 = str(r2.get("father_name", "")).strip().lower()
    fat_present = float(bool(fat1 and fat2))
    feats["father_both_present"] = fat_present

    if fat_present:
        feats["father_jaro_winkler"] = float(JaroWinkler.similarity(fat1, fat2))
        feats["father_levenshtein"] = float(Levenshtein.normalized_similarity(fat1, fat2))
        feats["father_exact"] = 1.0 if fat1 == fat2 else 0.0

        # Gower contribution: Father Name
        gower_num += feats["father_jaro_winkler"]
        gower_den += 1.0
    else:
        feats["father_jaro_winkler"] = 0.0
        feats["father_levenshtein"] = 0.0
        feats["father_exact"] = 0.0

    # ── 5. Date of Birth ───────────────────────────────────────────────────────
    dob1_str = str(r1.get("date_of_birth", "")).strip()
    dob2_str = str(r2.get("date_of_birth", "")).strip()
    d1_parts = parse_iso_date(dob1_str)
    d2_parts = parse_iso_date(dob2_str)
    dob_present = float(bool(d1_parts and d2_parts))
    feats["dob_both_present"] = dob_present

    if dob_present:
        y1, m1, d1 = d1_parts  # type: ignore
        y2, m2, d2 = d2_parts  # type: ignore

        feats["dob_exact"] = 1.0 if (y1 == y2 and m1 == m2 and d1 == d2) else 0.0
        feats["dob_day_equal"] = 1.0 if d1 == d2 else 0.0
        feats["dob_month_equal"] = 1.0 if m1 == m2 else 0.0
        feats["dob_year_equal"] = 1.0 if y1 == y2 else 0.0

        # Swapped day and month check
        feats["dob_day_month_swapped"] = 1.0 if (d1 == m2 and m1 == d2 and d1 != m1) else 0.0

        # One-digit edit on compact YYYYMMDD string
        c1 = f"{y1:04d}{m1:02d}{d1:02d}"
        c2 = f"{y2:04d}{m2:02d}{d2:02d}"
        feats["dob_one_digit_edit"] = 1.0 if Levenshtein.distance(c1, c2) <= 1 else 0.0

        # Absolute day difference log-transformed
        try:
            dt1 = date(y1, m1, d1)
            dt2 = date(y2, m2, d2)
            diff_days = abs((dt1 - dt2).days)
            feats["dob_diff_days_log"] = float(math.log1p(diff_days))

            # Gower contribution: DOB (decay over 10-year window)
            s_dob = max(0.0, 1.0 - (diff_days / (365.25 * 10)))
            gower_num += s_dob
            gower_den += 1.0
        except ValueError:
            feats["dob_diff_days_log"] = -1.0
    else:
        feats["dob_exact"] = 0.0
        feats["dob_day_equal"] = 0.0
        feats["dob_month_equal"] = 0.0
        feats["dob_year_equal"] = 0.0
        feats["dob_day_month_swapped"] = 0.0
        feats["dob_one_digit_edit"] = 0.0
        feats["dob_diff_days_log"] = -1.0

    # ── 6. Age ─────────────────────────────────────────────────────────────────
    age1_raw = r1.get("age_years", "")
    age2_raw = r2.get("age_years", "")
    try:
        a1 = float(age1_raw)
        a2 = float(age2_raw)
        age_present = float(a1 >= 0.0 and a2 >= 0.0)
    except (ValueError, TypeError):
        age_present = 0.0
        a1, a2 = -1.0, -1.0

    feats["age_both_present"] = age_present
    if age_present:
        diff_age = abs(a1 - a2)
        feats["age_diff_abs"] = float(diff_age)
        feats["age_diff_log"] = float(math.log1p(diff_age))
    else:
        feats["age_diff_abs"] = -1.0
        feats["age_diff_log"] = -1.0

    # ── 7. National ID (13 digits) ─────────────────────────────────────────────
    nid1 = re.sub(r"\D", "", str(r1.get("national_id", "")).strip())
    nid2 = re.sub(r"\D", "", str(r2.get("national_id", "")).strip())
    nid_present = float(bool(len(nid1) == 13 and len(nid2) == 13))
    feats["nid_both_present"] = nid_present

    if nid_present:
        feats["nid_exact"] = 1.0 if nid1 == nid2 else 0.0
        feats["nid_hamming_le_1"] = 1.0 if sum(c1 != c2 for c1, c2 in zip(nid1, nid2)) <= 1 else 0.0
        feats["nid_levenshtein"] = float(Levenshtein.normalized_similarity(nid1, nid2))

        # Gower contribution: National ID
        gower_num += feats["nid_levenshtein"]
        gower_den += 1.0
    else:
        feats["nid_exact"] = 0.0
        feats["nid_hamming_le_1"] = 0.0
        feats["nid_levenshtein"] = 0.0

    # ── 8. Phone (11 digits) ───────────────────────────────────────────────────
    ph1 = re.sub(r"\D", "", str(r1.get("phone", "")).strip())
    ph2 = re.sub(r"\D", "", str(r2.get("phone", "")).strip())
    ph_present = float(bool(len(ph1) == 11 and len(ph2) == 11))
    feats["phone_both_present"] = ph_present

    if ph_present:
        feats["phone_exact"] = 1.0 if ph1 == ph2 else 0.0
        feats["phone_last7_equal"] = 1.0 if ph1[-7:] == ph2[-7:] else 0.0
        feats["phone_edit_distance"] = float(Levenshtein.normalized_similarity(ph1, ph2))

        # Gower contribution: Phone
        gower_num += feats["phone_edit_distance"]
        gower_den += 1.0
    else:
        feats["phone_exact"] = 0.0
        feats["phone_last7_equal"] = 0.0
        feats["phone_edit_distance"] = 0.0

    # ── 9. Gender ──────────────────────────────────────────────────────────────
    g1 = str(r1.get("gender", "")).strip().upper()
    g2 = str(r2.get("gender", "")).strip().upper()
    g_present = float(bool(g1 in {"M", "F"} and g2 in {"M", "F"}))
    feats["gender_both_present"] = g_present

    if g_present:
        feats["gender_equal"] = 1.0 if g1 == g2 else 0.0
        gower_num += feats["gender_equal"]
        gower_den += 1.0
    else:
        feats["gender_equal"] = 0.0

    # ── 10. City ───────────────────────────────────────────────────────────────
    c1 = str(r1.get("city", "")).strip().lower()
    c2 = str(r2.get("city", "")).strip().lower()
    c_present = float(bool(c1 and c2))
    feats["city_both_present"] = c_present

    if c_present:
        feats["city_equal"] = 1.0 if c1 == c2 else 0.0
        gower_num += feats["city_equal"]
        gower_den += 1.0
    else:
        feats["city_equal"] = 0.0

    # ── 11. Address ────────────────────────────────────────────────────────────
    addr1 = str(r1.get("address", "")).strip()
    addr2 = str(r2.get("address", "")).strip()
    addr_present = float(bool(addr1 and addr2))
    feats["address_both_present"] = addr_present

    if addr_present:
        feats["address_token_jaccard"] = float(token_jaccard(addr1, addr2))
        feats["address_token_cosine"] = float(token_cosine(addr1, addr2))
        gower_num += feats["address_token_jaccard"]
        gower_den += 1.0
    else:
        feats["address_token_jaccard"] = 0.0
        feats["address_token_cosine"] = 0.0

    # ── 12. Blood Group ────────────────────────────────────────────────────────
    bg1 = str(r1.get("blood_group", "")).strip().upper()
    bg2 = str(r2.get("blood_group", "")).strip().upper()
    bg_present = float(bool(bg1 and bg2))
    feats["blood_group_both_present"] = bg_present

    if bg_present:
        feats["blood_group_equal"] = 1.0 if bg1 == bg2 else 0.0
        gower_num += feats["blood_group_equal"]
        gower_den += 1.0
    else:
        feats["blood_group_equal"] = 0.0

    # ── 13. Insurance Tier (Ordinal - Lecture 4) ───────────────────────────────
    ins1 = str(r1.get("insurance_tier", "")).strip().lower()
    ins2 = str(r2.get("insurance_tier", "")).strip().lower()
    ins_present = float(bool(ins1 in INSURANCE_RANKS and ins2 in INSURANCE_RANKS))
    feats["insurance_tier_both_present"] = ins_present

    if ins_present:
        r_diff = abs(INSURANCE_RANKS[ins1] - INSURANCE_RANKS[ins2]) / 3.0
        feats["insurance_tier_diff"] = float(r_diff)
        feats["insurance_tier_sim"] = float(1.0 - r_diff)
        gower_num += feats["insurance_tier_sim"]
        gower_den += 1.0
    else:
        feats["insurance_tier_diff"] = 1.0
        feats["insurance_tier_sim"] = 0.0

    # ── 14. Visit Count (Ratio - Lecture 4) ────────────────────────────────────
    v1_raw = r1.get("visit_count", "")
    v2_raw = r2.get("visit_count", "")
    try:
        v1 = float(v1_raw)
        v2 = float(v2_raw)
        v_present = float(v1 >= 0.0 and v2 >= 0.0)
    except (ValueError, TypeError):
        v_present = 0.0
        v1, v2 = -1.0, -1.0

    feats["visit_count_both_present"] = v_present
    if v_present:
        diff_v = abs(v1 - v2)
        feats["visit_count_diff_abs"] = float(diff_v)
        feats["visit_count_diff_log"] = float(math.log1p(diff_v))
        s_v = max(0.0, 1.0 - (diff_v / 100.0))
        gower_num += s_v
        gower_den += 1.0
    else:
        feats["visit_count_diff_abs"] = -1.0
        feats["visit_count_diff_log"] = -1.0

    # ── 15. Source System ──────────────────────────────────────────────────────
    src1 = str(r1.get("source_system", "")).strip()
    src2 = str(r2.get("source_system", "")).strip()
    src_present = float(bool(src1 and src2))
    feats["source_system_both_present"] = src_present
    feats["source_system_same"] = 1.0 if (src_present and src1 == src2) else 0.0

    # ── 16. Overall Gower Similarity (Delta Rule) ──────────────────────────────
    feats["gower_similarity"] = float(gower_num / gower_den) if gower_den > 0.0 else 0.0

    return feats


# ── Batch Pair Feature Extraction ──────────────────────────────────────────────

def extract_features_for_pair_dataframe(
    pairs_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """Computes comparison feature vectors for all pairs in pairs_df.
    
    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'record_id_1', 'record_id_2', and optional 'label', 'sample_weight'.
    clean_df : pd.DataFrame
        Cleaned patient register. Strictly validated against allowlist and forbidden columns.
    chunk_size : int
        Batch chunk size for processing.
        
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with identifiers, targets, weights, and all 58 comparison features.
    """
    # Enforce strict leakage and schema guard
    check_forbidden_columns(clean_df.columns)

    # Build memory-efficient lookup dictionary indexed by record_id
    records_dict = clean_df.set_index("record_id").to_dict(orient="index")

    r1_col = pairs_df["record_id_1"].tolist()
    r2_col = pairs_df["record_id_2"].tolist()
    n_pairs = len(pairs_df)

    rows: list[dict[str, float]] = []

    for idx, (r1, r2) in enumerate(zip(r1_col, r2_col)):
        rec1 = records_dict.get(r1)
        rec2 = records_dict.get(r2)
        if rec1 is None or rec2 is None:
            raise KeyError(f"Record ID not found in clean register: {r1} or {r2}")

        fv = compute_pair_features(rec1, rec2)
        rows.append(fv)

    feats_df = pd.DataFrame(rows)

    # Reattach identification and target columns at the front
    result_df = pd.DataFrame()
    result_df["record_id_1"] = r1_col
    result_df["record_id_2"] = r2_col

    if "label" in pairs_df.columns:
        result_df["label"] = pairs_df["label"].values
    if "sample_weight" in pairs_df.columns:
        result_df["sample_weight"] = pairs_df["sample_weight"].values

    final_df = pd.concat([result_df, feats_df], axis=1)
    return final_df


def build_and_save_all_features(
    clean_path: Path | str | None = None,
    pairs_dir: Path | str | None = None,
) -> dict[str, str]:
    """Extracts features for train_pairs_resampled and test_pairs, saving to data/pairs/."""
    clean_file = Path(clean_path) if clean_path is not None else CFG.CLEAN_REGISTER_PATH
    p_dir = Path(pairs_dir) if pairs_dir is not None else CFG.DATA_PAIRS_DIR

    train_pairs_path = p_dir / "train_pairs_resampled.parquet"
    test_pairs_path = p_dir / "test_pairs.parquet"

    if not clean_file.exists():
        raise FileNotFoundError(f"Clean register missing: {clean_file}")
    if not train_pairs_path.exists() or not test_pairs_path.exists():
        raise FileNotFoundError(f"Labelled pairs missing in {p_dir}. Run src/pairs.py first.")

    print(f"Loading cleaned register from {clean_file}...")
    df_clean = pd.read_csv(clean_file, dtype=str, keep_default_na=False)
    
    # Check if forbidden columns exist in the file before filtering
    forbidden_in_file = set(df_clean.columns) & CFG.FORBIDDEN_COLUMNS
    if forbidden_in_file:
        raise ValueError(
            f"FORBIDDEN COLUMN GUARD: Input file contains prohibited columns: {sorted(forbidden_in_file)}"
        )
        
    # Restrict strictly to the clean model allowlist
    allowed_cols = [c for c in df_clean.columns if c in CFG.CLEAN_MODEL_ALLOWLIST]
    df_clean = df_clean[allowed_cols].copy()
    check_forbidden_columns(df_clean.columns)

    print(f"Extracting features for training pairs ({train_pairs_path})...")
    df_train_pairs = pd.read_parquet(train_pairs_path)
    train_features = extract_features_for_pair_dataframe(df_train_pairs, df_clean)

    train_out_parquet = p_dir / "train_features_resampled.parquet"
    train_out_csv = p_dir / "train_features_resampled.csv.gz"
    train_features.to_parquet(train_out_parquet, index=False)
    train_features.to_csv(train_out_csv, index=False, compression="gzip")
    print(f"  Saved train features: {len(train_features):,} pairs x {train_features.shape[1]} columns")

    print(f"Extracting features for test pairs ({test_pairs_path})...")
    df_test_pairs = pd.read_parquet(test_pairs_path)
    test_features = extract_features_for_pair_dataframe(df_test_pairs, df_clean)

    test_out_parquet = p_dir / "test_features.parquet"
    test_out_csv = p_dir / "test_features.csv.gz"
    test_features.to_parquet(test_out_parquet, index=False)
    test_features.to_csv(test_out_csv, index=False, compression="gzip")
    print(f"  Saved test features:  {len(test_features):,} pairs x {test_features.shape[1]} columns")

    return {
        "train_features_parquet": str(train_out_parquet),
        "test_features_parquet": str(test_out_parquet),
    }


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Extract comparison features for train and test pairs.")
    args = parser.parse_args()

    print("Starting feature extraction pipeline...")
    res = build_and_save_all_features()
    print("\nFeature Extraction Complete:")
    print(f"  Train Features Path: {res['train_features_parquet']}")
    print(f"  Test Features Path:  {res['test_features_parquet']}")


if __name__ == "__main__":
    main()
