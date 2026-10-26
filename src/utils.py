"""Utility functions for metric persistence and report management.

Ensures that every reported number is dynamically recorded in reports/metrics.json
via non-destructive dictionary merging rather than hardcoding.
"""

import json
from pathlib import Path
from typing import Any

from src.config import METRICS_PATH


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges source dict into target dict.
    
    Nested dictionaries are merged rather than replaced so that existing metric
    sub-keys are preserved.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def load_metrics(metrics_file: Path | str | None = None) -> dict[str, Any]:
    """Loads metrics dictionary from reports/metrics.json (or specified file).
    
    Returns an empty dict if the file does not exist or contains invalid JSON.
    """
    target_path = Path(metrics_file) if metrics_file is not None else METRICS_PATH
    if not target_path.exists():
        return {}
    
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            data = json.loads(content)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_metrics(new_metrics: dict[str, Any], metrics_file: Path | str | None = None) -> dict[str, Any]:
    """Merges new_metrics into reports/metrics.json without overwriting unrelated keys.
    
    Parent directories are created automatically if they do not exist.
    Returns the updated full dictionary.
    """
    target_path = Path(metrics_file) if metrics_file is not None else METRICS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    current_metrics = load_metrics(target_path)
    updated_metrics = _deep_merge(current_metrics, new_metrics)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(updated_metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return updated_metrics


def save_metric(key: str, value: Any, metrics_file: Path | str | None = None) -> dict[str, Any]:
    """Convenience helper to merge a single top-level key/value into reports/metrics.json."""
    return save_metrics({key: value}, metrics_file=metrics_file)
