"""Carga, caché, partición y construcción de DataLoader del dataset.

Implementa carga con alcance de llamada y caché para el dataset wine_quality.
Nunca importa desde Experimentos/.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig, SplitSpec

URL_WINE_QUALITY_WHITE = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/wine-quality/"
    "winequality-white.csv"
)

# Prefijo + longitud del digest de la huella del dataset (derivada de SHA-256).
DATASET_VERSION_PREFIX = "sha256-"
_DATASET_VERSION_DIGEST_LENGTH = 16

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
DEFAULT_CACHE_SUBDIR = ".cache/prebenchmark"
DEFAULT_CACHE_FILENAME = "winequality-white.csv"


def fetch_wine_quality_csv(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Obtiene (o carga desde caché) el CSV de wine_quality.

    En la primera ejecución, se descarga desde UCI y se almacena en caché en cache_dir.
    """
    if cache_dir is None:
        cache_dir = Path.cwd() / DEFAULT_CACHE_SUBDIR
    else:
        cache_dir = Path(cache_dir) / DEFAULT_CACHE_SUBDIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / DEFAULT_CACHE_FILENAME

    if not cache_path.exists():
        df = pd.read_csv(URL_WINE_QUALITY_WHITE, sep=";")
        df.to_csv(cache_path, sep=";", index=False)

    return cache_path


def load_wine_quality(cache_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Carga el dataset wine_quality (alcance de llamada, con caché)."""
    csv_path = fetch_wine_quality_csv(cache_dir)
    return pd.read_csv(csv_path, sep=";")


def _format_fingerprint(digest: bytes) -> str:
    return f"{DATASET_VERSION_PREFIX}{digest.hex()[:_DATASET_VERSION_DIGEST_LENGTH]}"


def compute_file_fingerprint(path: str | os.PathLike[str]) -> str:
    """Devuelve una huella SHA-256 estable de los bytes del archivo del dataset.

    Detecta cualquier cambio en el archivo (un único valor editado altera el digest),
    por lo que se puede demostrar que dos ejecuciones usaron el mismo dataset subyacente.
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return _format_fingerprint(hasher.digest())


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Devuelve una huella SHA-256 estable de un dataset en memoria.

    Calcula el hash de la serialización CSV canónica separada por ``';'`` (el mismo
    formato en disco que el archivo de UCI), por lo que la huella es equivalente en
    contenido tanto si los datos provienen de un archivo como si se suministraron
    directamente (p. ej. pruebas sin conexión).
    """
    payload = df.to_csv(sep=";", index=False, lineterminator="\n").encode("utf-8")
    return _format_fingerprint(hashlib.sha256(payload).digest())


def resolve_dataset_version(df: pd.DataFrame | None = None) -> str:
    """Resuelve la cadena de versión del dataset para una ejecución.

    Con un ``df`` en memoria se usa directamente la huella de contenido; en caso
    contrario, se localiza el archivo del dataset en caché (descargándolo una vez si
    es necesario) y se calcula la huella de sus bytes. En ambos casos el resultado es
    un id ``sha256-<digest>``.
    """
    if df is not None:
        return compute_dataframe_fingerprint(df)
    return compute_file_fingerprint(fetch_wine_quality_csv())


def build_splits(
    cfg: ExperimentConfig | None = None,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Construye las particiones de entrenamiento/validación/prueba (60/20/20 por defecto, estratificadas)."""
    if df is None:
        df = load_wine_quality()

    if X is None or y is None:
        X = df.drop(columns=WINE_TARGET_COLUMN).values
        y_original = df[WINE_TARGET_COLUMN].values
        classes = np.sort(np.unique(y_original))
        class_to_index = {class_value: idx for idx, class_value in enumerate(classes)}
        y = np.array([class_to_index[v] for v in y_original])

    split = SplitSpec() if cfg is None else cfg.split
    train_size = split.train
    test_size = split.test
    val_size = split.val
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "split must sum to 1.0"

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        random_state=(cfg.seed if cfg else 42),
        stratify=y,
    )
    val_ratio_in_temp = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_ratio_in_temp,
        random_state=(cfg.seed if cfg else 42),
        stratify=y_temp,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def build_dataloaders(
    cfg: ExperimentConfig | None = None,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Construye los DataLoaders con batch_size=32 por defecto, con semilla."""
    if cfg is not None:
        batch_size = cfg.batch_size if cfg.batch_size else batch_size
        seed = cfg.seed if cfg.seed else seed

    splits = build_splits(cfg, X=X_train, y=y_train, df=df)
    if X_train is None:
        X_train, y_train = splits["X_train"], splits["y_train"]
    if X_val is None:
        X_val, y_val = splits["X_val"], splits["y_val"]
    if X_test is None:
        X_test, y_test = splits["X_test"], splits["y_test"]

    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train)
    X_val_t = scaler.transform(X_val) if X_val is not None else np.array([])
    X_test_t = scaler.transform(X_test) if X_test is not None else np.array([])

    train_dataset = TensorDataset(
        torch.tensor(X_train_t, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = (
        TensorDataset(
            torch.tensor(X_val_t, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        if len(X_val_t) > 0
        else None
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_t, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "n_train": len(X_train_t),
        "n_val": len(X_val_t) if len(X_val_t) > 0 else 0,
        "n_test": len(X_test_t),
    }
