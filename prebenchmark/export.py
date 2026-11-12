"""Exportaciones CSV de un experimento persistido.

Este módulo es una capa de *composición*: nunca reimplementa la persistencia,
la comparación, la agregación ni la lógica de decisiones. Lee ejecuciones, configuraciones y
decisiones a través de :class:`prebenchmark.store.ResultsStore`, normaliza los registros de
ejecución mediante la superficie pública de :mod:`prebenchmark.compare` (nunca tocando las claves
crudas ``state`` / ``execution_state`` / ``config``), y reutiliza
:func:`prebenchmark.compare.compare_to_baseline`,
:func:`prebenchmark.aggregate.aggregate_records`,
:func:`prebenchmark.decisions.decisions_to_records` para las tablas que estos poseen.

Salidas (bajo ``<out_dir>/<experiment_id>/``), un archivo por tipo de exportación,
con sufijo de marca de tiempo UTC para que las exportaciones repetidas nunca se sobrescriban entre sí::

    master_<ts>.csv              # cada ejecución (incl. FAILED) + métricas finales
    baseline_comparison_<ts>.csv # deltas frente a la línea base seleccionada
    aggregates_<ts>.csv          # estadísticas por grupo de repetición
    epochs_<ts>.csv              # historial plano por época
    configs_<ts>.csv             # definiciones de configuración + estado de validación
    decisions_<ts>.csv           # registro de decisiones manual de solo adición
    warnings_summary_<ts>.csv    # advertencias, fallos y ejecuciones PENDING_REVIEW

**Consistencia de N/A** -- cada valor faltante se escribe como el marcador literal
:data:`NA_MARKER` (``"N/A"``), nunca como un ``0`` fabricado. Esto se cumple para
deltas no comparables, métricas de validación ausentes, campos de configuración no coincidentes y
métricas de rendimiento de ejecuciones fallidas por igual.

**Los gráficos opcionales no bloquean** -- cuando ``include_plots=True`` se intenta
generar un gráfico de mejor esfuerzo; cualquier fallo (matplotlib ausente, backend sin interfaz gráfica, datos incorrectos)
se registra como una advertencia de exportación y los CSV se siguen produciendo.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aggregate import (
    AGGREGATE_METRICS,
    DEFAULT_MIN_REPETITIONS,
    GROUPING_MODES,
    aggregate_records,
    aggregate_runs,
)
from .compare import (
    DELTA_METRICS,
    VALIDATION_METRICS,
    BaselineSelection,
    ComparisonTable,
    compare_to_baseline,
    run_architecture,
    run_config,
    run_dataset,
    run_eval_set,
    run_failure_reason,
    run_field,
    run_metric_value,
    run_state,
    select_baseline,
)
from .decisions import decisions_to_records
from .identity import utc_timestamp
from .store import EPOCH_COLUMNS, EXPORTS_DIR_NAME, ResultsStore

__all__ = [
    "AGGREGATE_COLUMNS",
    "COMPARISON_COLUMNS",
    "CONFIG_COLUMNS",
    "DECISION_COLUMNS",
    "EPOCH_COLUMNS",
    "ExportResult",
    "MASTER_COLUMNS",
    "NA_MARKER",
    "WARNINGS_COLUMNS",
    "export_all",
    "persist_baseline",
]

# Único marcador documentado de valor faltante para cada exportación. Nunca se emite como 0.
NA_MARKER = "N/A"

# ---------------------------------------------------------------------------
# Definiciones de columnas estables (explícitas para que los encabezados nunca cambien, incluso con 0 filas)
# ---------------------------------------------------------------------------

_MASTER_METRICS: tuple[str, ...] = (
    tuple(m for m in DELTA_METRICS if m != "training_time") + VALIDATION_METRICS
)

MASTER_COLUMNS: tuple[str, ...] = (
    (
        "run_id",
        "config_id",
        "space_key",
        "experiment_id",
        "run_type",
        "dataset",
        "dataset_version",
        "architecture",
        "activation",
        "optimizer",
        "lr",
        "init",
        "loss_function",
        "batch_size",
        "seed",
        "split_train",
        "split_val",
        "split_test",
        "split_stratify",
        "max_epochs",
        "epochs_executed",
        "execution_state",
        "start_time",
        "end_time",
        "device",
        "training_time",
        "n_params",
        "evaluation_set",
    )
    + _MASTER_METRICS
    + (
        "warnings",
        "failure_reason",
    )
)

# Contexto de línea base (repetido por fila) + las columnas de la tabla comparadora.
COMPARISON_COLUMNS: tuple[str, ...] = (
    (
        "baseline_run_id",
        "baseline_config_id",
        "baseline_rationale",
        "run_id",
        "config_id",
        "state",
        "eval_set",
        "comparability_flag",
        "comparison_type",
        "informative_only",
        "controlled_variable",
        "varied_dimensions",
        "warnings",
        "failure_reason",
    )
    + DELTA_METRICS
    + tuple(f"delta_{m}" for m in DELTA_METRICS)
    + tuple(f"rel_{m}" for m in DELTA_METRICS)
)

_AGGREGATE_METRIC_SUFFIXES: tuple[str, ...] = ("n", "mean", "std", "min", "max", "range")

AGGREGATE_COLUMNS: tuple[str, ...] = (
    "group_key",
    "config_id",
    "config_ids",
    "space_key",
    "activation",
    "optimizer",
    "lr",
    "init",
    "dataset",
    "batch_size",
    "seeds",
    "run_ids",
    "n_total",
    "n_valid",
    "n_failed",
    "n_unstable",
    "failed_proportion",
    "unstable_proportion",
    "min_repetitions",
    "insufficient_repetitions",
    "evaluation_sets",
    "created_at",
) + tuple(
    f"{metric}_{suffix}" for metric in AGGREGATE_METRICS for suffix in _AGGREGATE_METRIC_SUFFIXES
)

CONFIG_COLUMNS: tuple[str, ...] = (
    "config_id",
    "space_key",
    "validation_state",
    "validation_rationale",
    "timestamp",
    "run_type",
    "dataset",
    "dataset_version",
    "architecture",
    "arch_input_dim",
    "hidden_layers",
    "arch_output_dim",
    "activation",
    "optimizer",
    "lr",
    "init",
    "loss",
    "batch_size",
    "max_epochs",
    "seed",
    "device",
    "split_train",
    "split_val",
    "split_test",
    "split_stratify",
    "scaler",
    "class_weight",
    "convergence_enabled",
    "convergence_metric",
    "convergence_patience",
    "convergence_min_delta",
    "convergence_mode",
    "config_json",
)

DECISION_COLUMNS: tuple[str, ...] = (
    "decision_id",
    "run_id",
    "config_id",
    "group_id",
    "decision_type",
    "label",
    "justification",
    "actor",
    "timestamp",
    "evidence",
)

WARNINGS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "config_id",
    "state",
    "category",
    "message",
    "epoch",
    "source",
)


# ---------------------------------------------------------------------------
# Resultado de exportación
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    """Resultado estructurado de :func:`export_all`."""

    experiment_id: str
    export_dir: Path
    timestamp: str
    files: Mapping[str, Path] = field(default_factory=dict)
    row_counts: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    baseline_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "export_dir": str(self.export_dir),
            "timestamp": self.timestamp,
            "files": {name: str(path) for name, path in self.files.items()},
            "row_counts": dict(self.row_counts),
            "warnings": list(self.warnings),
            "baseline_run_id": self.baseline_run_id,
        }


# ---------------------------------------------------------------------------
# Escritura CSV de bajo nivel (atómica; representación única de N/A)
# ---------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    """Convierte un valor para la salida CSV, mapeando ``None`` a :data:`NA_MARKER`."""
    if value is None:
        return NA_MARKER
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return value


def _write_csv(path: Path, columns: Sequence[str], records: Sequence[Mapping[str, Any]]) -> Path:
    """Escribe ``records`` en ``path`` con un encabezado fijo (atómico, sin pérdida por sobrescritura)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({column: _cell(record.get(column)) for column in columns})
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Constructores de filas (aplanadores delgados sobre la superficie de normalización compartida)
# ---------------------------------------------------------------------------


def _space_key_of(meta: Mapping[str, Any]) -> str | None:
    value = meta.get("space_key")
    if value:
        return str(value)
    config = run_config(meta)
    if not config:
        return None
    try:
        from .config import ExperimentConfig
        from .space import compute_space_key

        return compute_space_key(ExperimentConfig.from_dict(dict(config)))
    except Exception:  # pragma: no cover - defensivo: nunca hacer fallar una exportación
        return None


def _master_row(meta: Mapping[str, Any]) -> dict[str, Any]:
    dataset_name, dataset_version = run_dataset(meta)
    architecture = run_architecture(meta)
    split = run_field(meta, "split") or {}
    split = split if isinstance(split, Mapping) else {}
    warnings = meta.get("warnings") or []

    row: dict[str, Any] = {
        "run_id": meta.get("run_id"),
        "config_id": meta.get("config_id"),
        "space_key": _space_key_of(meta),
        "experiment_id": meta.get("experiment_id"),
        "run_type": run_field(meta, "run_type"),
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "architecture": architecture[0] if architecture else None,
        "activation": run_field(meta, "activation"),
        "optimizer": run_field(meta, "optimizer"),
        "lr": run_field(meta, "lr"),
        "init": run_field(meta, "init"),
        "loss_function": run_field(meta, "loss"),
        "batch_size": run_field(meta, "batch_size"),
        "seed": run_field(meta, "seed"),
        "split_train": split.get("train"),
        "split_val": split.get("val"),
        "split_test": split.get("test"),
        "split_stratify": split.get("stratify"),
        "max_epochs": run_field(meta, "max_epochs"),
        "epochs_executed": meta.get("epochs_executed"),
        "execution_state": run_state(meta),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "device": run_field(meta, "device"),
        "training_time": run_metric_value(meta, "training_time"),
        "n_params": run_field(meta, "n_params"),
        "evaluation_set": run_eval_set(meta),
        "warnings": "|".join(str(warning) for warning in warnings),
        "failure_reason": run_failure_reason(meta),
    }
    for metric in _MASTER_METRICS:
        row[metric] = run_metric_value(meta, metric)
    return row


def _comparison_records(table: ComparisonTable) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in table.to_records():
        record = dict(record)
        record["baseline_run_id"] = table.baseline_run_id
        record["baseline_config_id"] = table.baseline_config_id
        record["baseline_rationale"] = table.baseline_rationale
        records.append(record)
    return records


def _config_record(record: Mapping[str, Any]) -> dict[str, Any]:
    config = record.get("config") or {}
    preprocessing = record.get("preprocessing") or {}
    convergence = record.get("convergence") or {}
    dataset = config.get("dataset") or {}
    architecture = config.get("architecture") or {}
    split = config.get("split") or {}
    hidden = architecture.get("hidden_layers")
    rationale = record.get("validation_rationale") or []

    return {
        "config_id": record.get("config_id"),
        "space_key": record.get("space_key"),
        "validation_state": record.get("validation_state"),
        "validation_rationale": "|".join(str(item) for item in rationale),
        "timestamp": record.get("timestamp"),
        "run_type": config.get("run_type"),
        "dataset": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "architecture": architecture.get("name"),
        "arch_input_dim": architecture.get("input_dim"),
        "hidden_layers": "|".join(str(layer) for layer in hidden) if hidden else "",
        "arch_output_dim": architecture.get("output_dim"),
        "activation": config.get("activation"),
        "optimizer": config.get("optimizer"),
        "lr": config.get("lr"),
        "init": config.get("init"),
        "loss": config.get("loss"),
        "batch_size": config.get("batch_size"),
        "max_epochs": config.get("max_epochs"),
        "seed": config.get("seed"),
        "device": config.get("device"),
        "split_train": split.get("train"),
        "split_val": split.get("val"),
        "split_test": split.get("test"),
        "split_stratify": split.get("stratify"),
        "scaler": preprocessing.get("scaler"),
        "class_weight": preprocessing.get("class_weight"),
        "convergence_enabled": convergence.get("enabled"),
        "convergence_metric": convergence.get("metric"),
        "convergence_patience": convergence.get("patience"),
        "convergence_min_delta": convergence.get("min_delta"),
        "convergence_mode": convergence.get("mode"),
        "config_json": json.dumps(config, sort_keys=True, ensure_ascii=True, default=str),
    }


def _warning_records(
    runs: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in runs:
        state = run_state(meta)
        run_id = meta.get("run_id")
        config_id = meta.get("config_id")
        for warning in meta.get("warnings") or []:
            rows.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "state": state,
                    "category": "warning",
                    "message": str(warning),
                    "epoch": None,
                    "source": "run",
                }
            )
        if state == "FAILED":
            rows.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "state": state,
                    "category": "failure",
                    "message": run_failure_reason(meta) or "unknown failure",
                    "epoch": None,
                    "source": "run",
                }
            )
    for decision in decisions:
        if (decision.get("label") or "") != "PENDING_REVIEW":
            continue
        entity = decision.get("entity") or {}
        rows.append(
            {
                "run_id": entity.get("run_id", ""),
                "config_id": entity.get("config_id", ""),
                "state": "",
                "category": "pending_review",
                "message": decision.get("justification") or "",
                "epoch": None,
                "source": "decision",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Resolución de línea base + persistencia (diferida de PR4)
# ---------------------------------------------------------------------------


def _resolve_baseline(
    store: ResultsStore,
    experiment_id: str,
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline_run_id: str | None,
    baseline_config_id: str | None,
    use_documented_default: bool,
    timestamp: str,
    warnings: list[str],
) -> BaselineSelection | None:
    """Resuelve la línea base de forma explícita, a partir de metadatos persistidos, o no la resuelve en absoluto."""
    if baseline_run_id is not None or baseline_config_id is not None or use_documented_default:
        try:
            return select_baseline(
                runs,
                baseline_run_id=baseline_run_id,
                baseline_config_id=baseline_config_id,
                use_documented_default=use_documented_default,
                timestamp=timestamp,
            )
        except ValueError as error:
            warnings.append(f"baseline not selected: {error}")
            return None

    stored = store.load_experiment_metadata(experiment_id).get("baseline")
    if isinstance(stored, Mapping) and stored.get("run_id"):
        try:
            return select_baseline(
                runs,
                baseline_run_id=str(stored["run_id"]),
                rationale=stored.get("rationale") or None,
                timestamp=stored.get("timestamp") or timestamp,
            )
        except ValueError as error:
            warnings.append(f"stored baseline could not be resolved: {error}")
            return None
    return None


def persist_baseline(
    store: ResultsStore,
    experiment_id: str,
    selection: BaselineSelection,
) -> None:
    """Vuelca una selección de línea base en ``experiment.json`` y el registro de decisiones.

    La entrada de decisión se escribe como máximo una vez por (run_id, actor, type) para que
    las exportaciones repetidas no hagan crecer el registro de solo adición con duplicados.
    """
    metadata = store.load_experiment_metadata(experiment_id)
    metadata["baseline"] = {
        "run_id": selection.run_id,
        "config_id": selection.config_id,
        "state": selection.state,
        "rationale": selection.rationale,
        "timestamp": selection.timestamp,
    }
    store.write_experiment_metadata(experiment_id, metadata)

    for decision in store.load_decisions(experiment_id):
        entity = decision.get("entity") or {}
        if (
            decision.get("actor") == "export"
            and decision.get("decision_type") == "validate"
            and entity.get("run_id") == selection.run_id
        ):
            return
    store.append_decision(
        experiment_id,
        decision_type="validate",
        justification=selection.rationale,
        actor="export",
        run_id=selection.run_id,
        config_id=selection.config_id,
        label=None,
        evidence={"source": "export", "state": selection.state},
    )


# ---------------------------------------------------------------------------
# Gráficos opcionales (sin bloqueo)
# ---------------------------------------------------------------------------


def _write_plots(
    export_dir: Path,
    timestamp: str,
    master_records: Sequence[Mapping[str, Any]],
    epoch_records: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Gráficos de mejor esfuerzo. Lanza una excepción en caso de fallo; el llamador lo trata como opcional."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []

    if epoch_records:
        by_run: dict[str, list[Mapping[str, Any]]] = {}
        for record in epoch_records:
            by_run.setdefault(str(record.get("run_id")), []).append(record)
        figure, axis = plt.subplots()
        for run_id, rows in by_run.items():
            ordered = sorted(rows, key=lambda row: row.get("epoch") or 0)
            points = [
                (float(row.get("epoch") or 0), float(row["train_loss"]))
                for row in ordered
                if isinstance(row.get("train_loss"), (int, float))
            ]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=run_id,
            )
        axis.set_xlabel("epoch")
        axis.set_ylabel("train_loss")
        axis.set_title("Training loss per epoch")
        axis.legend(fontsize="x-small")
        path = export_dir / f"epoch_loss_{timestamp}.png"
        figure.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    if master_records:
        run_ids = [str(record.get("run_id")) for record in master_records]
        accuracies: list[float] = []
        for record in master_records:
            value = record.get("accuracy")
            accuracies.append(float(value) if isinstance(value, (int, float)) else 0.0)
        figure, axis = plt.subplots()
        axis.bar(range(len(run_ids)), accuracies)
        axis.set_xticks(range(len(run_ids)))
        axis.set_xticklabels(run_ids, rotation=90, fontsize="x-small")
        axis.set_ylabel("accuracy")
        axis.set_title("Final accuracy per run")
        path = export_dir / f"metric_accuracy_{timestamp}.png"
        figure.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------


def export_all(
    store: ResultsStore,
    experiment_id: str,
    out_dir: str | os.PathLike[str] | None = None,
    *,
    baseline_run_id: str | None = None,
    baseline_config_id: str | None = None,
    use_documented_default: bool = False,
    grouping: str = "config",
    min_repetitions: int = DEFAULT_MIN_REPETITIONS,
    include_plots: bool = True,
    epoch_run_ids: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> ExportResult:
    """Exporta cada tabla de resultados de ``experiment_id`` como CSV con marca de tiempo.

    ``out_dir`` es la raíz de exportaciones; los archivos se escriben en
    ``<out_dir>/<experiment_id>/``. Cuando se omite, por defecto es
    ``<store.root>/exports``. La línea base se resuelve de forma explícita, a partir de metadatos
    de experimento persistidos, o no se resuelve en absoluto -- nunca se adivina en silencio.
    """
    if grouping not in GROUPING_MODES:
        raise ValueError(f"grouping must be one of {GROUPING_MODES}, got {grouping!r}")
    if min_repetitions < 1:
        raise ValueError("min_repetitions must be >= 1")

    timestamp = timestamp or utc_timestamp()
    export_dir = (
        Path(out_dir) / experiment_id
        if out_dir is not None
        else Path(store.root) / EXPORTS_DIR_NAME / experiment_id
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    runs = store.query_runs(experiment_id)
    selection = _resolve_baseline(
        store,
        experiment_id,
        runs,
        baseline_run_id=baseline_run_id,
        baseline_config_id=baseline_config_id,
        use_documented_default=use_documented_default,
        timestamp=timestamp,
        warnings=warnings,
    )
    if selection is not None:
        persist_baseline(store, experiment_id, selection)
    else:
        warnings.append("baseline not selected; baseline_comparison exported without deltas")

    master_records = [_master_row(meta) for meta in runs]
    config_records = [_config_record(record) for record in store.load_configs(experiment_id)]
    decisions_raw = store.load_decisions(experiment_id)
    decision_records = decisions_to_records(list(decisions_raw))
    warning_records = _warning_records(runs, decisions_raw)

    epoch_records: list[dict[str, Any]] = []
    wanted = set(epoch_run_ids) if epoch_run_ids is not None else None
    for meta in runs:
        run_id = str(meta.get("run_id"))
        if wanted is not None and run_id not in wanted:
            continue
        try:
            loaded = store.load_run(run_id, experiment_id)
        except FileNotFoundError:  # pragma: no cover - defensivo
            continue
        for epoch in loaded.get("epochs", []):
            epoch_records.append(dict(epoch))
    epoch_records.sort(key=lambda row: (str(row.get("run_id")), row.get("epoch") or 0))

    if selection is not None:
        candidates = [meta for meta in runs if meta.get("run_id") != selection.run_id]
        table = compare_to_baseline(selection, candidates)
        comparison_records = _comparison_records(table)
    else:
        comparison_records = []

    groups = aggregate_runs(runs, grouping=grouping, min_repetitions=min_repetitions)
    aggregate_record_rows = aggregate_records(groups)

    tables: tuple[tuple[str, Sequence[str], Sequence[Mapping[str, Any]]], ...] = (
        ("master", MASTER_COLUMNS, master_records),
        ("baseline_comparison", COMPARISON_COLUMNS, comparison_records),
        ("aggregates", AGGREGATE_COLUMNS, aggregate_record_rows),
        ("epochs", EPOCH_COLUMNS, epoch_records),
        ("configs", CONFIG_COLUMNS, config_records),
        ("decisions", DECISION_COLUMNS, decision_records),
        ("warnings_summary", WARNINGS_COLUMNS, warning_records),
    )

    files: dict[str, Path] = {}
    row_counts: dict[str, int] = {}
    for name, columns, records in tables:
        files[name] = _write_csv(export_dir / f"{name}_{timestamp}.csv", columns, records)
        row_counts[name] = len(records)

    if include_plots:
        try:
            _write_plots(export_dir, timestamp, master_records, epoch_records)
        except Exception as error:  # noqa: BLE001 - los gráficos son estrictamente opcionales
            warnings.append(f"plot generation skipped: {error}")

    return ExportResult(
        experiment_id=experiment_id,
        export_dir=export_dir,
        timestamp=timestamp,
        files=files,
        row_counts=row_counts,
        warnings=tuple(warnings),
        baseline_run_id=selection.run_id if selection is not None else None,
    )
