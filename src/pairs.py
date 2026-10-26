"""Candidate pairs generation and ground-truth labelling module.

Runs the chosen multi-pass blocking scheme within the temporal train and test partitions,
labels pairs as match (1) or non-match (0) using ground truth, preserves natural proportions
for the test evaluation set, and builds an optional stratified re-sampled set for training
with exact importance re-weighting factors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config as CFG
from src.blocking import (
    pass_b1_soundex_dob_year,
    pass_b2_fn3_dob_year_month,
    pass_b3_phone_last7,
    pass_b4_national_id_exact,
    pass_b5_sorted_neighbourhood,
)
from src.utils import save_metric


def generate_partition_candidate_pairs(
    df_partition: pd.DataFrame,
    max_block_size: int = 250,
    window_size: int = 5,
) -> set[tuple[str, str]]:
    """Runs the 5-pass blocking scheme within a single partition (train or test).
    
    Returns the deduplicated union of candidate pairs (r1, r2) with r1 < r2.
    """
    p1, _, _, _ = pass_b1_soundex_dob_year(df_partition, max_block_size=max_block_size)
    p2, _, _, _ = pass_b2_fn3_dob_year_month(df_partition, max_block_size=max_block_size)
    p3, _, _, _ = pass_b3_phone_last7(df_partition, max_block_size=max_block_size)
    p4, _, _, _ = pass_b4_national_id_exact(df_partition, max_block_size=max_block_size)
    p5, _ = pass_b5_sorted_neighbourhood(df_partition, window_size=window_size)

    return p1 | p2 | p3 | p4 | p5


def label_pairs(
    pairs: set[tuple[str, str]],
    ground_truth_map: dict[str, str],
) -> pd.DataFrame:
    """Labels candidate pairs using ground-truth entity mapping.
    
    Parameters
    ----------
    pairs : set[tuple[str, str]]
        Set of candidate pairs (r1, r2).
    ground_truth_map : dict[str, str]
        Dictionary mapping record_id -> true_entity_id.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['record_id_1', 'record_id_2', 'label'].
    """
    sorted_pairs = sorted(pairs)
    r1_list = [p[0] for p in sorted_pairs]
    r2_list = [p[1] for p in sorted_pairs]

    labels = [
        1 if (ground_truth_map.get(r1) is not None and ground_truth_map.get(r1) == ground_truth_map.get(r2)) else 0
        for r1, r2 in zip(r1_list, r2_list)
    ]

    return pd.DataFrame({
        "record_id_1": r1_list,
        "record_id_2": r2_list,
        "label": labels,
    })


def create_stratified_resampled_train_pairs(
    df_train_pairs: pd.DataFrame,
    negative_ratio: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, float, float]:
    """Builds a stratified re-sampled training set keeping ALL positive pairs.
    
    Subsamples negative pairs to achieve the desired negative:positive ratio,
    and attaches exact inverse probability importance weights.
    
    Parameters
    ----------
    df_train_pairs : pd.DataFrame
        Full natural training candidate pairs.
    negative_ratio : float
        Desired number of negative pairs per positive pair (default: 5.0).
    seed : int
        Random seed for reproducible subsampling.
        
    Returns
    -------
    tuple[pd.DataFrame, float, float]
        (resampled_df, w_positive, w_negative)
    """
    positives = df_train_pairs[df_train_pairs["label"] == 1].copy()
    negatives = df_train_pairs[df_train_pairs["label"] == 0].copy()

    n_pos = len(positives)
    n_neg_pop = len(negatives)
    target_neg = int(round(n_pos * negative_ratio))
    target_neg = min(target_neg, n_neg_pop)

    sampled_negatives = negatives.sample(n=target_neg, random_state=seed).copy()

    # Importance re-weighting factors:
    # w_pos = 1.0 (all positives kept)
    # w_neg = N_neg_pop / N_neg_sample
    w_pos = 1.0
    w_neg = round(n_neg_pop / target_neg, 6) if target_neg > 0 else 1.0

    positives["sample_weight"] = w_pos
    sampled_negatives["sample_weight"] = w_neg

    resampled = pd.concat([positives, sampled_negatives], ignore_index=True)
    # Shuffle pairs deterministically
    resampled = resampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return resampled, w_pos, w_neg


def build_and_save_pairs(
    clean_path: Path | str | None = None,
    ground_truth_path: Path | str | None = None,
    splits_meta_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    negative_ratio: float = 5.0,
    max_block_size: int = 250,
    window_size: int = 5,
) -> dict[str, Any]:
    """Full workflow: loads split metadata, generates candidate pairs, labels, and saves."""
    clean_file = Path(clean_path) if clean_path is not None else CFG.CLEAN_REGISTER_PATH
    gt_file = Path(ground_truth_path) if ground_truth_path is not None else CFG.GROUND_TRUTH_PATH
    splits_file = (
        Path(splits_meta_path)
        if splits_meta_path is not None
        else (CFG.DATA_PAIRS_DIR / "entity_splits.json")
    )
    out_dir = Path(output_dir) if output_dir is not None else CFG.DATA_PAIRS_DIR

    if not clean_file.exists():
        raise FileNotFoundError(f"Clean register not found at {clean_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found at {gt_file}")
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits metadata not found at {splits_file}. Run src/split.py first.")

    with open(splits_file, encoding="utf-8") as f:
        split_data = json.load(f)

    record_to_side: dict[str, str] = split_data["record_to_side"]

    df_clean = pd.read_csv(clean_file, dtype=str, keep_default_na=False)
    df_gt = pd.read_csv(gt_file, dtype=str)
    gt_map = dict(zip(df_gt["record_id"], df_gt["true_entity_id"]))

    # Partition clean register into train and test
    df_clean["side"] = df_clean["record_id"].map(record_to_side)
    df_train = df_clean[df_clean["side"] == "train"].copy().drop(columns=["side"])
    df_test = df_clean[df_clean["side"] == "test"].copy().drop(columns=["side"])

    print(f"Generating candidate pairs for Train ({len(df_train):,} records)...")
    train_pairs_set = generate_partition_candidate_pairs(
        df_train, max_block_size=max_block_size, window_size=window_size
    )
    print(f"Generating candidate pairs for Test ({len(df_test):,} records)...")
    test_pairs_set = generate_partition_candidate_pairs(
        df_test, max_block_size=max_block_size, window_size=window_size
    )

    print("Labelling candidate pairs from ground truth...")
    df_train_natural = label_pairs(train_pairs_set, gt_map)
    df_train_natural["sample_weight"] = 1.0

    df_test_natural = label_pairs(test_pairs_set, gt_map)
    df_test_natural["sample_weight"] = 1.0

    # Build stratified resampled training set
    print(f"Building stratified resampled training set (neg_ratio={negative_ratio})...")
    df_train_resampled, w_pos, w_neg = create_stratified_resampled_train_pairs(
        df_train_natural, negative_ratio=negative_ratio, seed=CFG.RANDOM_SEED
    )

    # Save pair files in compressed formats (parquet and csv.gz)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_nat_parquet = out_dir / "train_pairs_natural.parquet"
    train_nat_csv = out_dir / "train_pairs_natural.csv.gz"
    train_res_parquet = out_dir / "train_pairs_resampled.parquet"
    train_res_csv = out_dir / "train_pairs_resampled.csv.gz"
    test_parquet = out_dir / "test_pairs.parquet"
    test_csv = out_dir / "test_pairs.csv.gz"

    df_train_natural.to_parquet(train_nat_parquet, index=False)
    df_train_natural.to_csv(train_nat_csv, index=False, compression="gzip")

    df_train_resampled.to_parquet(train_res_parquet, index=False)
    df_train_resampled.to_csv(train_res_csv, index=False, compression="gzip")

    df_test_natural.to_parquet(test_parquet, index=False)
    df_test_natural.to_csv(test_csv, index=False, compression="gzip")

    train_pos = int((df_train_natural["label"] == 1).sum())
    train_neg = int((df_train_natural["label"] == 0).sum())
    train_prev = train_pos / len(df_train_natural) if len(df_train_natural) > 0 else 0.0

    test_pos = int((df_test_natural["label"] == 1).sum())
    test_neg = int((df_test_natural["label"] == 0).sum())
    test_prev = test_pos / len(df_test_natural) if len(df_test_natural) > 0 else 0.0

    res_pos = int((df_train_resampled["label"] == 1).sum())
    res_neg = int((df_train_resampled["label"] == 0).sum())
    res_prev = res_pos / len(df_train_resampled) if len(df_train_resampled) > 0 else 0.0

    summary = {
        "train_natural_pairs": len(df_train_natural),
        "train_natural_positives": train_pos,
        "train_natural_negatives": train_neg,
        "train_natural_prevalence_pct": round(train_prev * 100, 4),
        "train_resampled_pairs": len(df_train_resampled),
        "train_resampled_positives": res_pos,
        "train_resampled_negatives": res_neg,
        "train_resampled_prevalence_pct": round(res_prev * 100, 4),
        "resample_weights": {
            "w_positive": w_pos,
            "w_negative": w_neg,
            "negative_to_positive_ratio": negative_ratio,
        },
        "test_natural_pairs": len(df_test_natural),
        "test_natural_positives": test_pos,
        "test_natural_negatives": test_neg,
        "test_natural_prevalence_pct": round(test_prev * 100, 4),
        "file_paths": {
            "train_natural": str(train_nat_parquet),
            "train_resampled": str(train_res_parquet),
            "test_natural": str(test_parquet),
        },
    }

    save_metric("pairs", summary)
    return summary


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Generate labelled candidate pairs for train and test.")
    parser.add_argument("--ratio", type=float, default=5.0, help="Negative to positive ratio for resampled training (default: 5.0)")
    args = parser.parse_args()

    print("Building labelled candidate pairs...")
    res = build_and_save_pairs(negative_ratio=args.ratio)
    print("\nPairs Generation Complete:")
    print(f"  Train Natural Pairs:   {res['train_natural_pairs']:,} (Positives: {res['train_natural_positives']:,}, Prev: {res['train_natural_prevalence_pct']}%)")
    print(f"  Train Resampled Pairs: {res['train_resampled_pairs']:,} (Positives: {res['train_resampled_positives']:,}, Prev: {res['train_resampled_prevalence_pct']}%)")
    print(f"    Resample weights:    w_pos={res['resample_weights']['w_positive']}, w_neg={res['resample_weights']['w_negative']}")
    print(f"  Test Natural Pairs:    {res['test_natural_pairs']:,} (Positives: {res['test_natural_positives']:,}, Prev: {res['test_natural_prevalence_pct']}%)")


if __name__ == "__main__":
    main()
