"""Capa de análisis estadístico sobre las exportaciones CSV de un experimento.

Este módulo consume el **contrato de exportación** (``master_*.csv``,
``configs_*.csv``, ``epochs_*.csv`` y, opcionalmente, ``aggregates_*.csv``) y
nunca lee el store interno. Esa decisión mantiene el análisis reproducible
sobre cualquier snapshot exportado y desacopla la estadística de la persistencia.

Responsabilidades:

* descubrir el snapshot más reciente de cada tabla y cargarlo como DataFrames;
* unir ``class_weight`` (ausente en ``master``) desde ``configs`` por ``config_id``;
* derivar métricas de convergencia desde ``epochs`` (``best_epoch``,
  ``time_to_best_epoch``, ``epoch_time_mean``);
* resumir cada configuración (media, desvío, IC 95 %, mediana, min/max);
* comparar cada configuración contra una referencia con tests de Welch y
  Mann-Whitney, IC 95 % de la diferencia y delta de Cliff;
* calcular efectos marginales por dimensión (activación, optimizador, lr,
  init, class_weight) con Kruskal-Wallis.

Los tests inferenciales se reportan como **evidencia**, no como veredicto: con
``n=5`` semillas por configuración la potencia es baja y las semillas no son
independientes entre configuraciones que comparten inicialización.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings as warnings_module
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CONVERGENCE_METRICS",
    "DIMENSION_COLUMNS",
    "EXPORT_TABLES",
    "METRIC_COLUMNS",
    "REQUIRED_TABLES",
    "AnalysisResult",
    "ExportBundle",
    "analyze_exports",
    "compare_to_reference",
    "derive_run_metrics",
    "discover_export_files",
    "load_export_bundle",
    "marginal_effects",
    "summarize_configurations",
]

# Tablas que el analizador sabe leer. ``aggregates`` es opcional: el resumen se
# recalcula desde las corridas para no depender de un agrupamiento previo.
EXPORT_TABLES: tuple[str, ...] = (
    "master",
    "configs",
    "epochs",
    "aggregates",
    "baseline_comparison",
    "decisions",
    "warnings_summary",
)
REQUIRED_TABLES: tuple[str, ...] = ("master", "configs", "epochs")

_FILE_PATTERN = re.compile(r"^(?P<table>[a-z_]+)_(?P<timestamp>\d{8}T\d{6}Z)\.csv$")

# Métricas que se resumen por configuración. ``best_epoch`` y
# ``time_to_best_epoch`` son derivadas de ``epochs``; el resto proviene de
# ``master``.
METRIC_COLUMNS: tuple[str, ...] = (
    "f1_macro",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "val_loss",
    "val_accuracy",
    "best_epoch",
    "epochs_executed",
    "time_to_best_epoch",
    "training_time",
)

# Métricas de convergencia que el reporte trata como ciudadanas de primera clase.
CONVERGENCE_METRICS: tuple[str, ...] = (
    "best_epoch",
    "epochs_executed",
    "time_to_best_epoch",
)

# Dimensiones experimentales que se comparan de forma marginal.
DIMENSION_COLUMNS: tuple[str, ...] = (
    "activation",
    "optimizer",
    "lr",
    "init",
    "class_weight",
)

_METRIC_LABELS: Mapping[str, str] = {
    "f1_macro": "F1 macro",
    "accuracy": "Exactitud",
    "balanced_accuracy": "Exactitud balanceada",
    "precision_macro": "Precisión macro",
    "recall_macro": "Recall macro",
    "val_loss": "Pérdida de validación",
    "val_accuracy": "Exactitud de validación",
    "best_epoch": "Mejor época",
    "epochs_executed": "Épocas ejecutadas",
    "time_to_best_epoch": "Tiempo hasta la mejor época (s)",
    "training_time": "Tiempo de entrenamiento (s)",
}

_SUMMARY_SUFFIXES: tuple[str, ...] = (
    "n",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "ci95_low",
    "ci95_high",
)

# Métricas que se comparan contra la referencia. Se limita el conjunto para
# evitar una tabla de comparaciones innecesariamente ruidosa.
_COMPARISON_METRICS: tuple[str, ...] = (
    "f1_macro",
    "time_to_best_epoch",
    "best_epoch",
    "training_time",
)

_MIN_VALID_RUNS = 3

# Mismo prefijo y longitud que ``prebenchmark.aggregate.compute_group_key`` para
# que el análisis hable el mismo idioma que la tabla de agregados exportada.
_GROUP_KEY_PREFIX = "grp-v1-"


def _group_key_from_config(config: Mapping[str, Any]) -> str:
    """Hash de la configuración normalizada por semilla (idéntico a ``aggregate``)."""
    normalized = {key: value for key, value in config.items() if key != "seed"}
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return f"{_GROUP_KEY_PREFIX}{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _fallback_group_key(row: Mapping[str, Any]) -> str:
    """Clave de respaldo cuando ``config_json`` no está disponible."""
    payload = {
        key: row.get(key)
        for key in (
            "space_key",
            "class_weight",
            "batch_size",
            "max_epochs",
            "dataset",
            "loss_function",
        )
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return f"grp-fallback-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Descubrimiento y carga del snapshot exportado
# ---------------------------------------------------------------------------


def discover_export_files(export_dir: str | Path) -> dict[str, Path]:
    """Devuelve la tabla más reciente de cada tipo dentro de ``export_dir``.

    Las exportaciones usan ``<tabla>_<UTC timestamp>.csv``. Si conviven varios
    snapshots, se elige el timestamp más alto por tabla. Las tablas requeridas
    deben existir; las opcionales se omiten en silencio.
    """
    directory = Path(export_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"directorio de exportación no encontrado: {directory}")

    latest: dict[str, tuple[str, Path]] = {}
    for path in directory.glob("*.csv"):
        match = _FILE_PATTERN.match(path.name)
        if match is None:
            continue
        table = match.group("table")
        timestamp = match.group("timestamp")
        if table not in EXPORT_TABLES:
            continue
        current = latest.get(table)
        if current is None or timestamp > current[0]:
            latest[table] = (timestamp, path)

    missing = [table for table in REQUIRED_TABLES if table not in latest]
    if missing:
        raise FileNotFoundError(
            f"al directorio de exportación {directory} le faltan tablas requeridas: {', '.join(missing)}"
        )
    return {table: path for table, (_, path) in sorted(latest.items())}


def _read_export(path: Path) -> pd.DataFrame:
    """Lee un CSV de exportación tratando el marcador literal ``N/A`` como faltante."""
    return pd.read_csv(path, na_values=["N/A"], keep_default_na=True)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _normalise_class_weight(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    if pd.isna(value):
        return "none"
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "n/a"}:
        return "none"
    return text


def _format_lr(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class ExportBundle:
    """Snapshot exportado ya cargado y normalizado."""

    experiment_id: str
    export_dir: Path
    timestamps: Mapping[str, str]
    files: Mapping[str, Path]
    runs: pd.DataFrame
    configs: pd.DataFrame
    epochs: pd.DataFrame
    aggregates: pd.DataFrame
    warnings: tuple[str, ...] = ()

    @property
    def timestamp(self) -> str:
        """Timestamp más reciente entre las tablas cargadas."""
        return max(self.timestamps.values()) if self.timestamps else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "export_dir": str(self.export_dir),
            "timestamps": dict(self.timestamps),
            "files": {name: str(path) for name, path in self.files.items()},
            "n_runs": int(len(self.runs)),
            "n_configs": int(self.runs["group_key"].nunique()) if "group_key" in self.runs else 0,
            "warnings": list(self.warnings),
        }


def load_export_bundle(export_dir: str | Path) -> ExportBundle:
    """Carga el snapshot exportado más reciente y lo normaliza.

    Normalizaciones aplicadas:

    * ``N/A`` se lee como faltante (no como cadena);
    * ``class_weight`` se une desde ``configs`` y se normaliza a ``"none"``;
    * las columnas numéricas se fuerzan a numérico;
    * se validan ``run_id``/``config_id`` y se registran advertencias de calidad.
    """
    directory = Path(export_dir)
    files = discover_export_files(directory)
    warnings: list[str] = []

    runs = _read_export(files["master"])
    configs = _read_export(files["configs"])
    epochs = _read_export(files["epochs"])
    aggregates = _read_export(files["aggregates"]) if "aggregates" in files else pd.DataFrame()

    if runs.empty:
        raise ValueError(f"la exportación master en {directory} no tiene filas")

    runs = _numeric(
        runs,
        (
            "lr",
            "batch_size",
            "seed",
            "max_epochs",
            "epochs_executed",
            "training_time",
            *METRIC_COLUMNS,
        ),
    )
    epochs = _numeric(
        epochs, ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")
    )
    configs = _numeric(
        configs, ("lr", "batch_size", "seed", "max_epochs", "arch_input_dim", "arch_output_dim")
    )

    for frame in (runs, configs, epochs):
        for column in ("run_id", "config_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")

    # ``class_weight`` no está en master: se incorpora desde configs por config_id.
    config_columns = ["config_id"]
    for column in ("class_weight", "validation_state", "hidden_layers", "space_key"):
        if column in configs.columns and column not in runs.columns:
            config_columns.append(column)
    if "class_weight" not in configs.columns:
        warnings.append(
            "la exportación de configs no tiene columna class_weight; todas las corridas se etiquetan como 'unknown'"
        )
        runs["class_weight"] = "unknown"
    else:
        configs = configs.copy()
        configs["class_weight"] = configs["class_weight"].map(_normalise_class_weight)
        runs = runs.merge(
            configs[config_columns].drop_duplicates(subset=["config_id"]),
            on="config_id",
            how="left",
            suffixes=("", "_config"),
        )
        runs["class_weight"] = runs["class_weight"].map(_normalise_class_weight)

    if "class_weight" not in runs.columns:
        runs["class_weight"] = "none"

    # La unidad de comparación es la configuración *sin semilla* (``group_key``).
    # ``config_id`` incluye la semilla, así que agrupar por él produciría una
    # "configuración" por corrida y destruiría toda la estadística.
    if "config_json" in configs.columns:
        key_by_config: dict[str, str] = {}
        for _, config_row in configs.drop_duplicates(subset=["config_id"]).iterrows():
            raw = config_row.get("config_json")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, Mapping):
                key_by_config[str(config_row["config_id"])] = _group_key_from_config(parsed)
        runs["group_key"] = runs["config_id"].astype(str).map(key_by_config)
        missing_keys = int(runs["group_key"].isna().sum())
        if missing_keys:
            warnings.append(
                f"{missing_keys} corrida(s) no tienen config_json parseable; se usó un group_key de respaldo"
            )
            fallback = runs.apply(lambda row: _fallback_group_key(row), axis=1)
            runs["group_key"] = runs["group_key"].fillna(fallback)
    else:
        warnings.append(
            "la exportación de configs no tiene config_json; group_key derivado de las dimensiones visibles"
        )
        runs["group_key"] = runs.apply(lambda row: _fallback_group_key(row), axis=1)

    # Dimensiones ausentes se rellenan para que el reporte nunca reciba NaN.
    for column in ("activation", "optimizer", "init", "dataset"):
        if column not in runs.columns:
            runs[column] = "unknown"
        runs[column] = runs[column].fillna("unknown").astype(str)
    runs["lr_label"] = runs["lr"].map(_format_lr) if "lr" in runs.columns else "unknown"

    if "execution_state" not in runs.columns:
        runs["execution_state"] = "UNKNOWN"
    runs["execution_state"] = runs["execution_state"].fillna("UNKNOWN").astype(str)
    runs["valid_run"] = runs["execution_state"].isin({"COMPLETED", "COMPLETED_WITH_WARNINGS"})

    missing_configs = sorted(set(runs["config_id"].dropna()) - set(configs["config_id"].dropna()))
    if missing_configs:
        warnings.append(
            f"{len(missing_configs)} config_id(s) de corridas no están en la exportación de configs: "
            + ", ".join(missing_configs[:5])
        )
    orphan_configs = sorted(set(configs["config_id"].dropna()) - set(runs["config_id"].dropna()))
    if orphan_configs:
        warnings.append(
            f"{len(orphan_configs)} configuración(es) registradas no tienen corridas: "
            + ", ".join(orphan_configs[:5])
        )

    duplicate_keys = runs.duplicated(subset=["group_key", "seed"], keep=False)
    if duplicate_keys.any():
        warnings.append(
            f"{int(duplicate_keys.sum())} corrida(s) repiten el mismo par (group_key, seed); "
            "tratarlas como verificaciones de reproducibilidad, no como muestras independientes"
        )

    timestamps = {
        table: _FILE_PATTERN.match(path.name).group("timestamp")  # type: ignore[union-attr]
        for table, path in files.items()
    }
    if len(set(timestamps.values())) > 1:
        warnings.append(
            "las tablas exportadas provienen de timestamps distintos: "
            + ", ".join(f"{table}={stamp}" for table, stamp in sorted(timestamps.items()))
        )

    return ExportBundle(
        experiment_id=directory.name,
        export_dir=directory,
        timestamps=timestamps,
        files=files,
        runs=runs.reset_index(drop=True),
        configs=configs.reset_index(drop=True),
        epochs=epochs.reset_index(drop=True),
        aggregates=aggregates,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Derivación de convergencia
# ---------------------------------------------------------------------------


def derive_run_metrics(bundle: ExportBundle) -> pd.DataFrame:
    """Añade métricas de convergencia derivadas de ``epochs`` a cada corrida.

    Definiciones (documentadas también en el reporte):

    * ``best_epoch``: época con menor ``val_loss`` (criterio de selección del
      mejor modelo, coherente con ``restore_best_weights``);
    * ``best_val_loss``: ese mínimo;
    * ``time_to_best_epoch``: suma acumulada de ``epoch_time`` hasta
      ``best_epoch`` (segundos de cómputo, sin overhead de arranque);
    * ``epoch_time_mean``: duración media por época;
    * ``epochs_recorded``: épocas efectivamente presentes en ``epochs``.
    """
    runs = bundle.runs.copy()
    epochs = bundle.epochs

    derived_columns = (
        "best_epoch",
        "best_val_loss",
        "time_to_best_epoch",
        "epoch_time_mean",
        "epochs_recorded",
        "final_val_loss",
    )
    for column in derived_columns:
        runs[column] = np.nan

    if epochs.empty:
        # La advertencia se emite en _quality_warnings a partir de las corridas sin historial.
        return runs

    required = {"run_id", "epoch", "val_loss", "epoch_time"}
    missing = required - set(epochs.columns)
    if missing:
        raise ValueError(
            f"a la exportación de epochs le faltan columnas requeridas: {', '.join(sorted(missing))}"
        )

    ordered = epochs.sort_values(["run_id", "epoch"], kind="stable")
    for run_id, group in ordered.groupby("run_id", sort=False):
        group = group.reset_index(drop=True)
        cumulative_time = group["epoch_time"].fillna(0.0).cumsum()
        best_position = group["val_loss"].idxmin() if group["val_loss"].notna().any() else None
        mask = runs["run_id"] == run_id
        if not mask.any():
            continue
        if best_position is None:
            runs.loc[mask, "epochs_recorded"] = int(len(group))
            runs.loc[mask, "epoch_time_mean"] = float(group["epoch_time"].mean())
            continue
        runs.loc[mask, "best_epoch"] = float(group.loc[best_position, "epoch"])
        runs.loc[mask, "best_val_loss"] = float(group.loc[best_position, "val_loss"])
        runs.loc[mask, "time_to_best_epoch"] = float(cumulative_time.loc[best_position])
        runs.loc[mask, "epoch_time_mean"] = float(group["epoch_time"].mean())
        runs.loc[mask, "epochs_recorded"] = int(len(group))
        runs.loc[mask, "final_val_loss"] = float(group["val_loss"].iloc[-1])

    return runs


# ---------------------------------------------------------------------------
# Estadística descriptiva e inferencial
# ---------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _mean_ci95(values: Iterable[Any]) -> tuple[float, float]:
    """IC 95 % de la media con distribución t de Student (n-1 grados de libertad)."""
    array = _finite(values)
    n = array.size
    if n < 2:
        return (math.nan, math.nan)
    from scipy import stats

    standard_error = float(array.std(ddof=1) / math.sqrt(n))
    if standard_error == 0.0:
        return (float(array.mean()), float(array.mean()))
    critical = float(stats.t.ppf(0.975, n - 1))
    mean = float(array.mean())
    return (mean - critical * standard_error, mean + critical * standard_error)


def _describe(values: Iterable[Any]) -> dict[str, float]:
    array = _finite(values)
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    low, high = _mean_ci95(array)
    return {
        "n": n,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if n > 1 else math.nan,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": low,
        "ci95_high": high,
    }


def _config_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('activation', '?')} | {row.get('optimizer', '?')} | "
        f"lr={_format_lr(row.get('lr'))} | {row.get('init', '?')} | "
        f"cw={row.get('class_weight', '?')}"
    )


def summarize_configurations(runs: pd.DataFrame) -> pd.DataFrame:
    """Resume cada configuración (``group_key``, sin semilla) con estadística descriptiva.

    Solo las corridas válidas (``COMPLETED``/``COMPLETED_WITH_WARNINGS``)
    alimentan las métricas; las fallidas se cuentan en ``n_failed``.
    """
    if runs.empty:
        return pd.DataFrame()

    dimension_columns = [column for column in DIMENSION_COLUMNS if column in runs.columns]
    rows: list[dict[str, Any]] = []
    for group_key, group in runs.groupby("group_key", sort=False, dropna=False):
        valid = group[group["valid_run"]]
        config_ids = sorted(str(value) for value in group["config_id"].dropna().unique())
        record: dict[str, Any] = {
            "group_key": str(group_key),
            "config_id": config_ids[0] if config_ids else "",
            "config_ids": "|".join(config_ids),
            "label": _config_label(group.iloc[0]),
            "space_key": group.iloc[0].get("space_key", ""),
            "dataset": group.iloc[0].get("dataset", ""),
            "batch_size": group.iloc[0].get("batch_size", math.nan),
            "hidden_layers": group.iloc[0].get("hidden_layers", ""),
            "validation_state": group.iloc[0].get("validation_state", ""),
            "lr_label": group.iloc[0].get("lr_label", _format_lr(group.iloc[0].get("lr"))),
            "n_runs": int(len(group)),
            "n_valid": int(len(valid)),
            "n_failed": int((~group["valid_run"]).sum()),
            "seeds": "|".join(
                str(int(seed))
                for seed in sorted(group["seed"].dropna().unique())
                if float(seed).is_integer()
            ),
        }
        for column in dimension_columns:
            record[column] = group.iloc[0].get(column)
        for metric in METRIC_COLUMNS:
            stats = _describe([]) if metric not in valid.columns else _describe(valid[metric])
            for suffix in _SUMMARY_SUFFIXES:
                record[f"{metric}_{suffix}"] = stats[suffix]
        rows.append(record)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["rank_f1"] = (
        summary["f1_macro_mean"].rank(ascending=False, method="min").astype("Int64")
    )
    summary["rank_time"] = (
        summary["time_to_best_epoch_mean"].rank(ascending=True, method="min").astype("Int64")
    )
    summary["insufficient_repetitions"] = summary["n_valid"] < _MIN_VALID_RUNS
    summary = summary.sort_values(
        ["f1_macro_mean", "time_to_best_epoch_mean", "group_key"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return summary


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    """Delta de Cliff: P(X>Y) - P(X<Y), en ``[-1, 1]``."""
    if first.size == 0 or second.size == 0:
        return math.nan
    greater = sum(float(np.sum(first > value)) for value in second)
    lesser = sum(float(np.sum(first < value)) for value in second)
    return (greater - lesser) / float(first.size * second.size)


def _effect_label(delta: float) -> str:
    if not math.isfinite(delta):
        return "n/d"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "despreciable"
    if magnitude < 0.33:
        return "pequeño"
    if magnitude < 0.474:
        return "medio"
    return "grande"


def _welch_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Diferencia de medias, IC 95 % de Welch, t y p bilateral."""
    from scipy import stats

    result = {
        "diff": math.nan,
        "ci95_low": math.nan,
        "ci95_high": math.nan,
        "welch_t": math.nan,
        "welch_p": math.nan,
    }
    if first.size < 2 or second.size < 2:
        return result

    mean_first = float(first.mean())
    mean_second = float(second.mean())
    var_first = float(first.var(ddof=1))
    var_second = float(second.var(ddof=1))
    diff = mean_first - mean_second
    result["diff"] = diff

    standard_error_squared = var_first / first.size + var_second / second.size
    if standard_error_squared <= 0.0:
        result["ci95_low"] = diff
        result["ci95_high"] = diff
        return result

    standard_error = math.sqrt(standard_error_squared)
    numerator = standard_error_squared**2
    denominator = 0.0
    if first.size > 1 and var_first > 0:
        denominator += (var_first / first.size) ** 2 / (first.size - 1)
    if second.size > 1 and var_second > 0:
        denominator += (var_second / second.size) ** 2 / (second.size - 1)
    degrees = numerator / denominator if denominator > 0 else min(first.size, second.size) - 1
    critical = float(stats.t.ppf(0.975, degrees))
    result["ci95_low"] = diff - critical * standard_error
    result["ci95_high"] = diff + critical * standard_error

    with np.errstate(invalid="ignore", divide="ignore"), warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore", RuntimeWarning)
        statistic, pvalue = stats.ttest_ind(first, second, equal_var=False, nan_policy="omit")
    if np.isfinite(statistic):
        result["welch_t"] = float(statistic)
    if np.isfinite(pvalue):
        result["welch_p"] = float(pvalue)
    return result


def _mann_whitney(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    if first.size < 3 or second.size < 3:
        return (math.nan, math.nan)
    if (
        np.allclose(first, first[0])
        and np.allclose(second, second[0])
        and math.isclose(first[0], second[0])
    ):
        return (math.nan, math.nan)
    from scipy import stats

    try:
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("ignore", RuntimeWarning)
            statistic, pvalue = stats.mannwhitneyu(first, second, alternative="two-sided")
    except ValueError:
        return (math.nan, math.nan)
    return (float(statistic), float(pvalue))


def select_reference(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    reference_config_id: str | None = None,
) -> tuple[str, str, str]:
    """Selecciona la configuración de referencia y documenta el motivo.

    ``reference_config_id`` acepta tanto un ``group_key`` como cualquier
    ``config_id`` individual; en el segundo caso se resuelve al grupo que lo
    contiene. Si no se indica referencia, se usa la mejor F1 media (desempate
    por menor tiempo hasta el mejor epoch y menor desvío). Es una referencia
    **empírica**, no una línea base experimental pre-registrada.
    """
    if summary.empty:
        raise ValueError(
            "no se puede seleccionar una referencia desde un resumen de configuraciones vacío"
        )
    if summary["f1_macro_mean"].notna().sum() == 0:
        raise ValueError(
            "ninguna configuración tiene un F1 macro medio finito; no se puede seleccionar una referencia"
        )

    if reference_config_id is not None:
        match = summary[summary["group_key"] == reference_config_id]
        if match.empty:
            containing = runs[runs["config_id"].astype(str) == reference_config_id]
            if not containing.empty:
                resolved = str(containing.iloc[0]["group_key"])
                match = summary[summary["group_key"] == resolved]
        if match.empty:
            raise ValueError(
                f"la referencia {reference_config_id!r} no es un group_key ni un config_id de la exportación"
            )
        row = match.iloc[0]
        return (
            str(row["group_key"]),
            str(row["label"]),
            f"referencia explícita indicada por quien llama ({reference_config_id})",
        )

    ordered = summary.sort_values(
        ["f1_macro_mean", "time_to_best_epoch_mean", "f1_macro_std", "group_key"],
        ascending=[False, True, True, True],
        na_position="last",
        kind="stable",
    )
    row = ordered.iloc[0]
    return (
        str(row["group_key"]),
        str(row["label"]),
        "referencia empírica: mayor F1 macro medio (desempate: menor tiempo hasta la mejor época, menor desvío)",
    )


def compare_to_reference(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    reference_group_key: str,
) -> pd.DataFrame:
    """Compara cada configuración contra la referencia para las métricas clave.

    Cada fila reporta la diferencia de medias, su IC 95 % de Welch, el test t de
    Welch, Mann-Whitney U y el delta de Cliff con su magnitud.
    """
    if summary.empty:
        return pd.DataFrame()

    reference_runs = runs[(runs["group_key"] == reference_group_key) & runs["valid_run"]]
    if reference_runs.empty:
        raise ValueError(
            f"el group_key de referencia {reference_group_key!r} no tiene corridas válidas"
        )

    rows: list[dict[str, Any]] = []
    for _, config in summary.iterrows():
        group_key = str(config["group_key"])
        if group_key == reference_group_key:
            continue
        candidate_runs = runs[(runs["group_key"] == group_key) & runs["valid_run"]]
        for metric in _COMPARISON_METRICS:
            if metric not in runs.columns:
                continue
            first = _finite(candidate_runs[metric])
            second = _finite(reference_runs[metric])
            welch = _welch_difference(first, second)
            u_statistic, u_pvalue = _mann_whitney(first, second)
            delta = _cliffs_delta(first, second)
            rows.append(
                {
                    "group_key": group_key,
                    "config_id": str(config.get("config_id", "")),
                    "label": str(config["label"]),
                    "reference_group_key": reference_group_key,
                    "metric": metric,
                    "metric_label": _METRIC_LABELS.get(metric, metric),
                    "n_config": int(first.size),
                    "n_reference": int(second.size),
                    "mean_config": float(first.mean()) if first.size else math.nan,
                    "mean_reference": float(second.mean()) if second.size else math.nan,
                    "diff": welch["diff"],
                    "ci95_low": welch["ci95_low"],
                    "ci95_high": welch["ci95_high"],
                    "welch_t": welch["welch_t"],
                    "welch_p": welch["welch_p"],
                    "mannwhitney_u": u_statistic,
                    "mannwhitney_p": u_pvalue,
                    "cliffs_delta": delta,
                    "effect_size": _effect_label(delta),
                }
            )
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons = comparisons.sort_values(
            ["metric", "diff", "group_key"],
            ascending=[True, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return comparisons


def marginal_effects(
    runs: pd.DataFrame,
    dimensions: Sequence[str] = DIMENSION_COLUMNS,
) -> pd.DataFrame:
    """Resume cada nivel de cada dimensión y calcula Kruskal-Wallis entre niveles.

    Es un análisis **marginal**: mezcla configuraciones y semillas, por lo que
    describe tendencias, no efectos causales aislados.
    """
    valid = runs[runs["valid_run"]]
    if valid.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in valid.columns:
            continue
        levels = valid[dimension].dropna().astype(str).unique().tolist()
        samples = [
            _finite(valid.loc[valid[dimension].astype(str) == level, "f1_macro"])
            for level in levels
        ]
        samples = [sample for sample in samples if sample.size > 0]
        pvalue = math.nan
        if len(samples) >= 2:
            from scipy import stats

            try:
                with warnings_module.catch_warnings():
                    warnings_module.simplefilter("ignore", RuntimeWarning)
                    _, pvalue = stats.kruskal(*samples)
            except ValueError:
                pvalue = math.nan
        for level in sorted(levels):
            subset = valid[valid[dimension].astype(str) == level]
            f1 = _describe(subset["f1_macro"])
            time_stats = _describe(subset["time_to_best_epoch"])
            epoch_stats = _describe(subset["best_epoch"])
            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "n_runs": int(len(subset)),
                    "n_configs": int(subset["group_key"].nunique()),
                    "f1_mean": f1["mean"],
                    "f1_std": f1["std"],
                    "f1_ci95_low": f1["ci95_low"],
                    "f1_ci95_high": f1["ci95_high"],
                    "time_mean": time_stats["mean"],
                    "time_std": time_stats["std"],
                    "best_epoch_mean": epoch_stats["mean"],
                    "kruskal_p": float(pvalue) if math.isfinite(pvalue) else math.nan,
                }
            )
    marginals = pd.DataFrame(rows)
    if not marginals.empty:
        marginals = marginals.sort_values(
            ["dimension", "f1_mean", "level"],
            ascending=[True, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return marginals


# ---------------------------------------------------------------------------
# Resultado agregado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    """Resultado completo del análisis de un snapshot exportado."""

    bundle: ExportBundle
    runs: pd.DataFrame
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    marginals: pd.DataFrame
    reference_group_key: str
    reference_label: str
    reference_reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experiment_id(self) -> str:
        return self.bundle.experiment_id

    @property
    def best_config(self) -> Mapping[str, Any]:
        return self.summary.iloc[0].to_dict() if not self.summary.empty else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "export_dir": str(self.bundle.export_dir),
            "timestamp": self.bundle.timestamp,
            "n_runs": int(len(self.runs)),
            "n_configs": int(self.summary.shape[0]),
            "reference_group_key": self.reference_group_key,
            "reference_label": self.reference_label,
            "reference_reason": self.reference_reason,
            "warnings": list(self.warnings),
        }


def _quality_warnings(bundle: ExportBundle, runs: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    warnings = list(bundle.warnings)
    failed = int((~runs["valid_run"]).sum())
    if failed:
        warnings.append(
            f"{failed} corrida(s) no están en COMPLETED/COMPLETED_WITH_WARNINGS y fueron excluidas de las métricas"
        )

    if not bundle.epochs.empty:
        recorded = runs["epochs_recorded"].fillna(0)
        no_history = int((recorded == 0).sum())
        if no_history:
            warnings.append(
                f"{no_history} corrida(s) no tienen historial de épocas; faltan métricas de convergencia"
            )
        mismatch = runs[
            runs["epochs_recorded"].notna()
            & runs["epochs_executed"].notna()
            & (runs["epochs_recorded"] != runs["epochs_executed"])
        ]
        if not mismatch.empty:
            warnings.append(
                f"{len(mismatch)} corrida(s) tienen epochs_recorded != epochs_executed; "
                "revisar si la exportación de épocas está truncada"
            )

    insufficient = summary[summary["insufficient_repetitions"]]
    if not insufficient.empty:
        warnings.append(
            f"{len(insufficient)} configuración(es) tienen menos de {_MIN_VALID_RUNS} repeticiones válidas"
        )

    if "class_weight" in runs.columns:
        values = set(runs["class_weight"].astype(str).unique())
        if values == {"none", "balanced"}:
            warnings.append(
                "class_weight divide la exportación en dos variantes experimentales; "
                "el reporte las muestra por separado y nunca las mezcla en silencio"
            )
    return warnings


def analyze_exports(
    export_dir: str | Path,
    *,
    reference_config_id: str | None = None,
) -> AnalysisResult:
    """Carga un snapshot exportado y produce el análisis completo."""
    bundle = load_export_bundle(export_dir)
    runs = derive_run_metrics(bundle)
    summary = summarize_configurations(runs)
    if summary.empty:
        raise ValueError(f"no se pudo resumir ninguna configuración desde {export_dir}")
    reference_id, reference_label, reference_reason = select_reference(
        summary, runs, reference_config_id=reference_config_id
    )
    comparisons = compare_to_reference(runs, summary, reference_id)
    marginals = marginal_effects(runs)
    warnings = _quality_warnings(bundle, runs, summary)
    return AnalysisResult(
        bundle=bundle,
        runs=runs,
        summary=summary,
        comparisons=comparisons,
        marginals=marginals,
        reference_group_key=reference_id,
        reference_label=reference_label,
        reference_reason=reference_reason,
        warnings=tuple(dict.fromkeys(warnings)),
    )
