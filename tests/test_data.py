"""Tests for data module."""

from __future__ import annotations

import pandas as pd

from prebenchmark import config, data


def test_load_wine_quality_uses_cache(tmp_path, monkeypatch):
    import numpy as np
    from tests.conftest import WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN, WINE_CLASSES

    rng = np.random.default_rng(42)
    n = 140
    d = {c: rng.normal(0, 1, n) for c in WINE_FEATURE_COLUMNS}
    d["free sulfur dioxide"] = rng.integers(0, 80, n).astype(float)
    d["total sulfur dioxide"] = rng.integers(0, 250, n).astype(float)
    d[WINE_TARGET_COLUMN] = np.array([WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n)])
    df = pd.DataFrame(d)
    cache = tmp_path / "winequality-white.csv"
    df.to_csv(cache, sep=";", index=False)
    from pathlib import Path

    monkeypatch.setattr(data, "fetch_wine_quality_csv", lambda cache_dir=None: Path(cache))
    loaded = data.load_wine_quality(cache_dir=tmp_path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape[0] == df.shape[0]


def test_build_splits_60_20_20_defaults():
    import numpy as np
    from tests.conftest import WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN, WINE_CLASSES
    rng = np.random.default_rng(42)
    n = 140
    d = {c: rng.normal(0, 1, n) for c in WINE_FEATURE_COLUMNS}
    d["free sulfur dioxide"] = rng.integers(0, 80, n).astype(float)
    d["total sulfur dioxide"] = rng.integers(0, 250, n).astype(float)
    d[WINE_TARGET_COLUMN] = np.array([WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n)])
    df = pd.DataFrame(d)
    splits = data.build_splits(config.ExperimentConfig(), df=df)
    total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert total == len(df)


def test_build_dataloaders_batch_size_32():
    import numpy as np
    from tests.conftest import WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN, WINE_CLASSES
    rng = np.random.default_rng(42)
    n = 140
    d = {c: rng.normal(0, 1, n) for c in WINE_FEATURE_COLUMNS}
    d["free sulfur dioxide"] = rng.integers(0, 80, n).astype(float)
    d["total sulfur dioxide"] = rng.integers(0, 250, n).astype(float)
    d[WINE_TARGET_COLUMN] = np.array([WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n)])
    df = pd.DataFrame(d)
    dls = data.build_dataloaders(config.ExperimentConfig(), df=df)
    assert dls["train_loader"].batch_size == 32


def _sample_df(n: int = 40, seed: int = 7) -> pd.DataFrame:
    import numpy as np
    from tests.conftest import WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN, WINE_CLASSES

    rng = np.random.default_rng(seed)
    d = {c: rng.normal(0, 1, n) for c in WINE_FEATURE_COLUMNS}
    d["free sulfur dioxide"] = rng.integers(0, 80, n).astype(float)
    d["total sulfur dioxide"] = rng.integers(0, 250, n).astype(float)
    d[WINE_TARGET_COLUMN] = np.array([WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n)])
    return pd.DataFrame(d)


def test_dataframe_fingerprint_is_stable_and_prefixed():
    df = _sample_df()
    first = data.compute_dataframe_fingerprint(df)
    second = data.compute_dataframe_fingerprint(df.copy())
    assert first == second
    assert first.startswith(data.DATASET_VERSION_PREFIX)


def test_dataframe_fingerprint_changes_when_content_changes():
    df = _sample_df()
    original = data.compute_dataframe_fingerprint(df)
    changed = df.copy()
    changed.iloc[0, 0] = changed.iloc[0, 0] + 1.0
    assert data.compute_dataframe_fingerprint(changed) != original


def test_resolve_dataset_version_uses_dataframe_when_given():
    df = _sample_df()
    assert data.resolve_dataset_version(df) == data.compute_dataframe_fingerprint(df)


def test_file_fingerprint_matches_dataframe_fingerprint(tmp_path):
    df = _sample_df()
    path = tmp_path / "winequality-white.csv"
    df.to_csv(path, sep=";", index=False, lineterminator="\n")
    assert data.compute_file_fingerprint(path) == data.compute_dataframe_fingerprint(df)
