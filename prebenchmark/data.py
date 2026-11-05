"""Carga, caché, partición y construcción de DataLoader del dataset.

Implementa un registro de datasets con alcance de llamada y caché en disco.
El dataset por defecto sigue siendo ``wine_quality`` (UCI), pero la carga, el
target y la resolución de la versión son genéricos: cualquier entrada de
:data:`DATASETS` puede usarse como ``cfg.dataset.name``.

Los datasets se cachean como CSV con separador ``';'`` y saltos de línea ``\\n``
para que la huella SHA-256 sea estable entre sistemas. Nunca importa desde
``Experimentos/``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULT_DATASET, ExperimentConfig, SplitSpec

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

# Columna objetivo normalizada para los datasets que no son wine_quality.
GENERIC_TARGET_COLUMN = "target"


@dataclass(frozen=True)
class DatasetDefinition:
    """Metadatos y loader de un dataset registrado.

    ``loader`` devuelve un ``DataFrame`` con una columna objetivo llamada
    ``target`` (los datasets de scikit-learn/OpenML se normalizan y codifican al
    cargarse). ``cache_filename`` es el nombre del CSV canónico en caché.
    """

    name: str
    source: str
    target: str
    cache_filename: str
    loader: Callable[[], pd.DataFrame]
    description: str = ""


@dataclass(frozen=True)
class DatasetInfo:
    """Resumen derivado del dataset cargado (dimensiones y distribución)."""

    name: str
    source: str
    target: str
    n_samples: int
    n_features: int
    n_classes: int
    cache_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "cache_path": str(self.cache_path),
        }


def _canonicalize_generic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el target de un dataset genérico a códigos ``0..k-1``.

    También elimina filas sin target. La codificación ordenada mantiene el mismo
    mapeo entre ejecuciones, de modo que la partición estratificada y las
    métricas son reproducibles.
    """
    if GENERIC_TARGET_COLUMN not in df.columns:
        return df
    frame = df.dropna(subset=[GENERIC_TARGET_COLUMN]).reset_index(drop=True)
    codes, _ = pd.factorize(frame[GENERIC_TARGET_COLUMN], sort=True)
    frame = frame.copy()
    frame[GENERIC_TARGET_COLUMN] = codes
    return frame


def _load_bundled(loader_name: str) -> pd.DataFrame:
    """Carga un dataset empaquetado de scikit-learn (p. ej. ``load_digits``)."""
    from sklearn import datasets

    bunch = getattr(datasets, loader_name)()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame[GENERIC_TARGET_COLUMN] = y
    return _canonicalize_generic_frame(frame)


def _load_openml(openml_name: str, version: int = 1) -> pd.DataFrame:
    """Carga un dataset de OpenML y conserva solo las features numéricas.

    Los datasets registrados (satimage, vehicle, waveform-5000, segment) son
    completamente numéricos; el filtro evita que una columna categórica
    accidental rompa ``StandardScaler``.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        name=openml_name,
        version=version,
        as_frame=True,
        parser="auto",
    )
    features = bunch.data.select_dtypes(include=[np.number]).reset_index(drop=True)
    frame = features.copy()
    frame[GENERIC_TARGET_COLUMN] = pd.Series(np.asarray(bunch.target)).reset_index(drop=True)
    return _canonicalize_generic_frame(frame)


def _load_wine_quality() -> pd.DataFrame:
    return pd.read_csv(fetch_wine_quality_csv(), sep=";")


DATASETS: dict[str, DatasetDefinition] = {
    "wine_quality": DatasetDefinition(
        name="wine_quality",
        source="UCI winequality-white.csv",
        target=WINE_TARGET_COLUMN,
        cache_filename=DEFAULT_CACHE_FILENAME,
        loader=_load_wine_quality,
        description="Vinos blancos, 7 clases ordinales con desbalance extremo (ratio 439x).",
    ),
    "satimage": DatasetDefinition(
        name="satimage",
        source="OpenML satimage v1",
        target=GENERIC_TARGET_COLUMN,
        cache_filename="satimage.csv",
        loader=lambda: _load_openml("satimage"),
        description="Imágenes Landsat, 6 clases, 6.430 muestras, ratio 2,45.",
    ),
    "vehicle": DatasetDefinition(
        name="vehicle",
        source="OpenML vehicle v1 (Statlog)",
        target=GENERIC_TARGET_COLUMN,
        cache_filename="vehicle.csv",
        loader=lambda: _load_openml("vehicle"),
        description="Siluetas de vehículos, 4 clases, 846 muestras, casi balanceado.",
    ),
    "waveform_5000": DatasetDefinition(
        name="waveform_5000",
        source="OpenML waveform-5000 v1",
        target=GENERIC_TARGET_COLUMN,
        cache_filename="waveform-5000.csv",
        loader=lambda: _load_openml("waveform-5000"),
        description="Señales sintéticas con ruido, 3 clases balanceadas, 5.000 muestras.",
    ),
    "segment": DatasetDefinition(
        name="segment",
        source="OpenML segment v1",
        target=GENERIC_TARGET_COLUMN,
        cache_filename="segment.csv",
        loader=lambda: _load_openml("segment"),
        description="Segmentación de imágenes, 7 clases perfectamente balanceadas; techo alto.",
    ),
    "digits": DatasetDefinition(
        name="digits",
        source="scikit-learn load_digits",
        target=GENERIC_TARGET_COLUMN,
        cache_filename="digits.csv",
        loader=lambda: _load_bundled("load_digits"),
        description="Dígitos 8x8, 10 clases balanceadas; dataset de control fácil.",
    ),
}


def available_datasets() -> tuple[str, ...]:
    """Nombres de los datasets registrados, en orden estable."""
    return tuple(DATASETS)


def get_dataset(name: str | None = None) -> DatasetDefinition:
    """Devuelve la definición registrada o lanza ``ValueError`` con las opciones."""
    resolved = name or DEFAULT_DATASET
    try:
        return DATASETS[resolved]
    except KeyError as error:
        options = ", ".join(available_datasets())
        raise ValueError(
            f"unknown dataset '{resolved}'; available datasets: {options}"
        ) from error


def _resolve_cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    base = Path.cwd() if cache_dir is None else Path(cache_dir)
    cache_root = base / DEFAULT_CACHE_SUBDIR
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


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


def fetch_dataset_csv(
    name: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Devuelve el CSV canónico del dataset, cargándolo y cacheándolo si hace falta.

    ``wine_quality`` conserva su descarga UCI histórica; el resto se escribe con
    saltos de línea ``\\n`` para que la huella sea estable entre plataformas.
    """
    definition = get_dataset(name)
    if definition.name == "wine_quality":
        return fetch_wine_quality_csv(cache_dir)

    cache_root = _resolve_cache_root(cache_dir)
    cache_path = cache_root / definition.cache_filename
    if not cache_path.exists():
        frame = definition.loader()
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        frame.to_csv(tmp_path, sep=";", index=False, lineterminator="\n")
        os.replace(tmp_path, cache_path)
    return cache_path


def load_dataset(
    name: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Carga un dataset registrado (con caché en disco)."""
    return pd.read_csv(fetch_dataset_csv(name, cache_dir), sep=";")


def load_wine_quality(cache_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Carga el dataset wine_quality (alcance de llamada, con caché)."""
    return load_dataset("wine_quality", cache_dir)


def describe_dataset(
    name: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> DatasetInfo:
    """Carga el dataset y deriva sus dimensiones reales.

    Se usa para construir el espacio de configuraciones: ``n_features`` y
    ``n_classes`` nunca se codifican a mano para un dataset nuevo.
    """
    definition = get_dataset(name)
    cache_path = fetch_dataset_csv(definition.name, cache_dir)
    frame = pd.read_csv(cache_path, sep=";")
    if definition.target not in frame.columns:
        raise ValueError(
            f"dataset '{definition.name}' has no target column '{definition.target}'"
        )
    feature_count = frame.drop(columns=[definition.target]).shape[1]
    class_count = int(frame[definition.target].nunique())
    return DatasetInfo(
        name=definition.name,
        source=definition.source,
        target=definition.target,
        n_samples=len(frame),
        n_features=feature_count,
        n_classes=class_count,
        cache_path=cache_path,
    )


def dataset_base_config(
    name: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> ExperimentConfig:
    """Construye la configuración base (protocolo) para un dataset registrado.

    Mantiene las cuatro dimensiones libres en sus valores de línea base y deriva
    ``input_dim``/``output_dim`` de los datos reales.
    """
    from .config import (
        BASELINE_ACTIVATION,
        BASELINE_INIT,
        BASELINE_LR,
        BASELINE_OPTIMIZER,
        ArchitectureSpec,
        DatasetSpec,
    )

    info = describe_dataset(name, cache_dir)
    return ExperimentConfig(
        dataset=DatasetSpec(
            name=info.name,
            source=info.source,
            target=info.target,
            n_features=info.n_features,
        ),
        architecture=ArchitectureSpec(
            name=f"{info.name}_mlp",
            input_dim=info.n_features,
            hidden_layers=[32, 16],
            output_dim=info.n_classes,
        ),
        activation=BASELINE_ACTIVATION,
        optimizer=BASELINE_OPTIMIZER,
        lr=BASELINE_LR,
        init=BASELINE_INIT,
    )


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


def resolve_dataset_version(
    df: pd.DataFrame | None = None,
    dataset: str | None = None,
) -> str:
    """Resuelve la cadena de versión del dataset para una ejecución.

    Con un ``df`` en memoria se usa directamente la huella de contenido; en caso
    contrario, se localiza el CSV canónico del dataset registrado en caché
    (cargándolo una vez si es necesario) y se calcula la huella de sus bytes. En
    ambos casos el resultado es un id ``sha256-<digest>``.
    """
    if df is not None:
        return compute_dataframe_fingerprint(df)
    return compute_file_fingerprint(fetch_dataset_csv(dataset))


def build_splits(
    cfg: ExperimentConfig | None = None,
    *,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Construye las particiones de entrenamiento/validación/prueba (60/20/20 por defecto, estratificadas)."""
    target_column = (
        cfg.dataset.target if cfg is not None else WINE_TARGET_COLUMN
    )
    if df is None and (X is None or y is None):
        dataset_name = cfg.dataset.name if cfg is not None else DEFAULT_DATASET
        df = load_dataset(dataset_name)

    if X is None or y is None:
        if df is None or target_column not in df.columns:
            raise ValueError(
                f"target column '{target_column}' not found in the provided dataset"
            )
        X = df.drop(columns=target_column).values
        y_original = df[target_column].values
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
