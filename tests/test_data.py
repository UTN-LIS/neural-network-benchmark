"""Tests for data module."""

from __future__ import annotations

import pandas as pd
from prebenchmark import config, data


def test_load_wine_quality_uses_cache(tmp_path, monkeypatch):
    import numpy as np
    from tests.conftest import WINE_CLASSES, WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN

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
    from tests.conftest import WINE_CLASSES, WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN

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
    from tests.conftest import WINE_CLASSES, WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN

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
    from tests.conftest import WINE_CLASSES, WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN

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


def _toy_definition() -> data.DatasetDefinition:
    import numpy as np
    import pandas as pd

    def loader() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x0": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                "x1": np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
                "target": np.array([0, 1, 0, 1, 2, 2]),
            }
        )

    return data.DatasetDefinition(
        name="toy",
        source="in-memory test fixture",
        target=data.GENERIC_TARGET_COLUMN,
        cache_filename="toy.csv",
        loader=loader,
    )


def test_dataset_registry_matches_known_datasets():
    from prebenchmark.config import KNOWN_DATASETS

    assert set(data.DATASETS) == set(KNOWN_DATASETS)


def test_registered_dataset_is_cached_and_described(tmp_path, monkeypatch):
    definition = _toy_definition()
    calls = []

    def counting_loader():
        calls.append(1)
        return definition.loader()

    monkeypatch.setitem(
        data.DATASETS,
        "toy",
        data.DatasetDefinition(
            name=definition.name,
            source=definition.source,
            target=definition.target,
            cache_filename=definition.cache_filename,
            loader=counting_loader,
        ),
    )

    first = data.fetch_dataset_csv("toy", cache_dir=tmp_path)
    second = data.fetch_dataset_csv("toy", cache_dir=tmp_path)
    assert first == second
    assert len(calls) == 1

    loaded = data.load_dataset("toy", cache_dir=tmp_path)
    assert loaded.shape == (6, 3)

    info = data.describe_dataset("toy", cache_dir=tmp_path)
    assert info.n_samples == 6
    assert info.n_features == 2
    assert info.n_classes == 3
    assert info.target == data.GENERIC_TARGET_COLUMN


def test_dataset_base_config_derives_dimensions(tmp_path, monkeypatch):
    monkeypatch.setitem(data.DATASETS, "toy", _toy_definition())

    base = data.dataset_base_config("toy", cache_dir=tmp_path)
    assert base.dataset.name == "toy"
    assert base.dataset.target == data.GENERIC_TARGET_COLUMN
    assert base.dataset.n_features == 2
    assert base.architecture.input_dim == 2
    assert base.architecture.output_dim == 3
    assert base.architecture.hidden_layers == [32, 16]


def test_build_splits_uses_configured_target_column(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    monkeypatch.setitem(data.DATASETS, "toy", _toy_definition())
    cfg = data.dataset_base_config("toy", cache_dir=tmp_path)
    cfg.split = config.SplitSpec(train=0.5, val=0.25, test=0.25)

    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "target": np.array([i % 3 for i in range(n)]),
        }
    )
    splits = data.build_splits(cfg, df=df)
    assert len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"]) == n
    assert set(np.unique(splits["y_train"])) <= {0, 1, 2}


def test_default_space_for_wine_quality_keeps_historic_dimensions():
    from prebenchmark import space

    base = space.default_space("wine_quality").base
    assert base.dataset.n_features == 11
    assert base.architecture.input_dim == 11
    assert base.architecture.output_dim == 7
    assert base.dataset.target == "quality"
