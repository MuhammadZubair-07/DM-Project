"""Trivial smoke test verifying required imports and configuration."""

import pytest


def test_required_third_party_imports():
    """Verify all pinned dependencies import without error."""
    import pandas
    import numpy
    import sklearn
    import rapidfuzz
    import streamlit
    import plotly
    import matplotlib
    import joblib
    import pytest
    import scipy
    import pyarrow

    assert pandas.__name__ == "pandas"
    assert numpy.__name__ == "numpy"
    assert sklearn.__name__ == "sklearn"
    assert rapidfuzz.__name__ == "rapidfuzz"
    assert streamlit.__name__ == "streamlit"
    assert plotly.__name__ == "plotly"
    assert matplotlib.__name__ == "matplotlib"
    assert joblib.__name__ == "joblib"
    assert pytest.__name__ == "pytest"
    assert scipy.__name__ == "scipy"
    assert pyarrow.__name__ == "pyarrow"


def test_internal_module_imports():
    """Verify src.config and src.utils import cleanly and provide expected constants."""
    from src import config
    from src import utils

    assert config.ROOT_DIR.exists()
    assert config.RANDOM_SEED == 42
    assert isinstance(config.CANONICAL_CITIES, list)
    assert len(config.CANONICAL_CITIES) > 0
    assert isinstance(config.SENTINEL_DATES, set)

    # Test load/save metrics non-destructive merge using an isolated temp path
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_metrics_file = Path(tmp_dir) / "test_metrics.json"
        
        # Initial save
        utils.save_metric("phase_0", {"status": "setup_complete"}, metrics_file=tmp_metrics_file)
        loaded = utils.load_metrics(tmp_metrics_file)
        assert loaded["phase_0"]["status"] == "setup_complete"

        # Merge new key without losing existing
        utils.save_metric("test_key", 123, metrics_file=tmp_metrics_file)
        loaded_updated = utils.load_metrics(tmp_metrics_file)
        assert loaded_updated["phase_0"]["status"] == "setup_complete"
        assert loaded_updated["test_key"] == 123
