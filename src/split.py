"""Entity-grouped temporal splitting module.

Assigns each true entity to train or test based on its EARLIEST registration date
so that train comprises roughly the earlier 70% of the timeline. All records belonging
to an entity are strictly assigned to the same partition, preventing entity leakage.

Ground truth (true_entity_id) is used solely for this partitioning and for pair labelling,
never as an input feature for candidate comparison or model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as CFG
from src.utils import save_metric


def compute_temporal_entity_split(
    clean_path: Path | str | None = None,
    ground_truth_path: Path | str | None = None,
    train_entity_ratio: float = 0.70,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Computes entity-grouped temporal split based on earliest registration date.
    
    Parameters
    ----------
    clean_path : Path | str | None
        Path to register_clean.csv.
    ground_truth_path : Path | str | None
        Path to ground_truth.csv.
    train_entity_ratio : float
        Target proportion of entities assigned to the earlier train partition (~0.70).
    output_dir : Path | str | None
        Directory where entity_splits.json will be saved.
        
    Returns
    -------
    dict[str, Any]
        Summary metadata including cutoff_date, entity/record counts, and partition maps.
    """
    clean_file = Path(clean_path) if clean_path is not None else CFG.CLEAN_REGISTER_PATH
    gt_file = Path(ground_truth_path) if ground_truth_path is not None else CFG.GROUND_TRUTH_PATH
    out_dir = Path(output_dir) if output_dir is not None else CFG.DATA_PAIRS_DIR

    if not clean_file.exists():
        raise FileNotFoundError(f"Clean register not found at {clean_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found at {gt_file}")

    df_clean = pd.read_csv(clean_file, usecols=["record_id", "registration_date"], dtype=str)
    df_gt = pd.read_csv(gt_file, dtype=str)

    # Merge on record_id to connect registration dates with true entities
    merged = df_clean.merge(df_gt, on="record_id", how="inner")
    if len(merged) != len(df_clean):
        raise ValueError(f"Ground truth record count ({len(df_gt)}) does not match clean register ({len(df_clean)})")

    # For each true entity, find its EARLIEST registration date across all encounters
    entity_min_dates = (
        merged.groupby("true_entity_id")["registration_date"]
        .min()
        .sort_values(kind="mergesort")
    )
    total_entities = len(entity_min_dates)

    # Determine cutoff date for earlier ~70% of entities
    target_idx = int(round(total_entities * train_entity_ratio))
    target_idx = min(max(0, target_idx), total_entities - 1)
    cutoff_date = str(entity_min_dates.iloc[target_idx])

    # Assign entities: <= cutoff_date -> train, > cutoff_date -> test
    train_entities = set(entity_min_dates[entity_min_dates <= cutoff_date].index)
    test_entities = set(entity_min_dates[entity_min_dates > cutoff_date].index)

    # Rigorous leakage assertions
    assert len(train_entities & test_entities) == 0, "FATAL: Entity overlap detected between train and test!"
    assert len(train_entities) + len(test_entities) == total_entities, "Entity partition count mismatch"

    # Map record_id to partition side
    merged["side"] = merged["true_entity_id"].apply(
        lambda e: "train" if e in train_entities else "test"
    )

    train_records = set(merged[merged["side"] == "train"]["record_id"])
    test_records = set(merged[merged["side"] == "test"]["record_id"])

    assert len(train_records & test_records) == 0, "Record overlap detected between train and test"
    assert len(train_records) + len(test_records) == len(df_clean), "Record partition count mismatch"

    # Save entity-to-side mapping
    entity_split_map = {e: "train" for e in train_entities}
    entity_split_map.update({e: "test" for e in test_entities})

    record_split_map = {r: "train" for r in train_records}
    record_split_map.update({r: "test" for r in test_records})

    out_dir.mkdir(parents=True, exist_ok=True)
    split_meta_path = out_dir / "entity_splits.json"
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cutoff_date": cutoff_date,
                "total_entities": total_entities,
                "train_entities_count": len(train_entities),
                "test_entities_count": len(test_entities),
                "train_entity_ratio": round(len(train_entities) / total_entities, 4),
                "test_entity_ratio": round(len(test_entities) / total_entities, 4),
                "total_records": len(df_clean),
                "train_records_count": len(train_records),
                "test_records_count": len(test_records),
                "train_record_ratio": round(len(train_records) / len(df_clean), 4),
                "test_record_ratio": round(len(test_records) / len(df_clean), 4),
                "entity_to_side": entity_split_map,
                "record_to_side": record_split_map,
            },
            f,
            indent=2,
        )

    summary = {
        "cutoff_date": cutoff_date,
        "total_entities": total_entities,
        "train_entities": len(train_entities),
        "test_entities": len(test_entities),
        "train_entity_pct": round(len(train_entities) / total_entities * 100, 2),
        "test_entity_pct": round(len(test_entities) / total_entities * 100, 2),
        "total_records": len(df_clean),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_record_pct": round(len(train_records) / len(df_clean) * 100, 2),
        "test_record_pct": round(len(test_records) / len(df_clean) * 100, 2),
        "split_meta_path": str(split_meta_path),
    }

    # Non-destructive persistence into reports/metrics.json
    save_metric("split", summary)

    return summary


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Run entity-grouped temporal splitting.")
    parser.add_argument("--ratio", type=float, default=0.70, help="Train entity ratio (default: 0.70)")
    args = parser.parse_args()

    print(f"Computing entity-grouped temporal split (target ratio: {args.ratio})...")
    res = compute_temporal_entity_split(train_entity_ratio=args.ratio)
    print("\nEntity Temporal Split Complete:")
    print(f"  Cutoff Date:         {res['cutoff_date']}")
    print(f"  Train Entities:      {res['train_entities']:,} ({res['train_entity_pct']}%)")
    print(f"  Test Entities:       {res['test_entities']:,} ({res['test_entity_pct']}%)")
    print(f"  Train Records:       {res['train_records']:,} ({res['train_record_pct']}%)")
    print(f"  Test Records:        {res['test_records']:,} ({res['test_record_pct']}%)")
    print(f"  Saved metadata to:   {res['split_meta_path']}")


if __name__ == "__main__":
    main()
