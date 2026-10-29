"""Shared pytest fixtures.

Everything here is offline-safe: no network access, and GPU execution is forced
off so results are deterministic on any machine. The synthetic wine_quality
fixture mirrors the UCI winequality-white shape (11 features + ``quality``
target with classes 3..9) and is reused by later slices' integration tests.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Keep CUDA invisible to any test that imports torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# UCI winequality-white feature columns, in on-disk order.
WINE_FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
WINE_TARGET_COLUMN = "quality"
WINE_CLASSES = (3, 4, 5, 6, 7, 8, 9)


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CPU execution for any test that happens to use torch."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for unit tests
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)


@pytest.fixture
def wine_quality_df() -> pd.DataFrame:
    """Offline synthetic wine_quality-shaped DataFrame (7 classes)."""
    rng = np.random.default_rng(42)
    n_samples = 140
    data: dict[str, np.ndarray] = {
        column: rng.normal(loc=0.0, scale=1.0, size=n_samples)
        for column in WINE_FEATURE_COLUMNS
    }
    # Give a couple of columns realistic integer-ish ranges.
    data["free sulfur dioxide"] = rng.integers(0, 80, size=n_samples).astype(float)
    data["total sulfur dioxide"] = rng.integers(0, 250, size=n_samples).astype(float)
    # Cycle through classes 3..9 so every class is present for stratification.
    data[WINE_TARGET_COLUMN] = np.array(
        [WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n_samples)]
    )
    return pd.DataFrame(data)


@pytest.fixture
def wine_quality_csv(wine_quality_df: pd.DataFrame, tmp_path) -> "os.PathLike[str]":
    """Write the synthetic dataset as the UCI ';'-separated CSV to a temp path."""
    path = tmp_path / "winequality-white.csv"
    wine_quality_df.to_csv(path, sep=";", index=False)
    return path
