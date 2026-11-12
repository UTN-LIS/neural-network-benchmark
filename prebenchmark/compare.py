"""Comparación con una línea base para las ejecuciones registradas.

Funciones puras sobre registros de metadatos de ejecución (tal como los devuelve
:meth:`prebenchmark.store.ResultsStore.query_runs`): sin E/S, sin torch, por lo
que el módulo puede importarse de forma segura desde cualquier sitio.

Semántica (decisiones vinculantes para este fragmento):

* **Línea base explícita** -- :func:`select_baseline` nunca realiza una
  suposición. Genera una excepción a menos que el llamador indique una
  ejecución (``baseline_run_id``), una configuración (``baseline_config_id``) o
  opte explícitamente por la *configuración predeterminada documentada*
  (``use_documented_default=True``: ReLU + Adam + lr 0.001 + He +
  CrossEntropyLoss + arquitectura ``wine_quality`` -- la única configuración que
  se ha ejecutado en este repositorio). Cada selección registra su justificación
  y una marca de tiempo UTC.
* **Diferencias (deltas)** -- diferencia absoluta ``candidato - línea base``
  para el perfil completo de métricas (exactitud, exactitud balanceada,
  precisión/exhaustividad/F1 macro, pérdida, tiempo de entrenamiento). Las
  diferencias relativas son opcionales y solo se emiten cuando el denominador
  de la línea base es distinto de cero y finito; la base se indica en
  :data:`RELATIVE_BASIS`.
* **Guardas de comparabilidad** -- se comparan primero el dataset (nombre +
  versión), la arquitectura (nombre, input_dim, hidden_layers, output_dim), las
  proporciones y estratificación de la partición, max_epochs, la configuración
  de convergencia, run_type y el conjunto de evaluación. Cualquier discrepancia
  impide calcular diferencias de métricas: la fila se marca como
  ``non_comparable`` y todas las diferencias son ``None`` -- nunca se emite un
  valor ficticio.
* **Controlado frente a multivariable** -- se comparan las dimensiones abiertas
  (:data:`prebenchmark.space.SPACE_DIMENSIONS`: activación/optimizador/lr/
  inicialización) más ``class_weight``. Exactamente una diferencia -> ``controlled``
  (se indica la variable de interés). Más de una diferencia -> ``multi_variable`` y la fila
  se marca como ``informative_only``: las diferencias siguen mostrándose, pero
  no pueden atribuirse a una única causa. ``class_weight`` no divide el espacio de 36,
  pero sí cambia el protocolo de entrenamiento: dos ejecuciones que solo difieren en el
  balanceo son una comparación controlada, no dos ejecuciones idénticas.
* **Ejecuciones fallidas** -- se enumeran con su motivo de fallo, se marcan
  como no comparables y nunca aportan diferencias.
* **Sin puntuación compuesta** -- se conserva el perfil completo de métricas y
  sus compromisos; no se calcula ningún orden único ponderado.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import (
    BASELINE_ACTIVATION,
    BASELINE_INIT,
    BASELINE_LR,
    BASELINE_OPTIMIZER,
    ExperimentConfig,
)
from .identity import utc_timestamp
from .space import SPACE_DIMENSIONS

__all__ = [
    "COMPARISON_TYPES",
    "DELTA_METRICS",
    "RELATIVE_BASIS",
    "VALIDATION_METRICS",
    "BaselineSelection",
    "ComparabilityResult",
    "ComparisonRow",
    "ComparisonTable",
    "check_comparability",
    "compare_to_baseline",
    "documented_baseline_config",
    "filter_runs",
    "run_architecture",
    "run_class_weight",
    "run_config",
    "run_dataset",
    "run_eval_set",
    "run_failure_reason",
    "run_field",
    "run_metric_value",
    "run_state",
    "select_baseline",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Métricas que reciben una diferencia absoluta (candidato - línea base, con signo).
# ``training_time`` se encuentra en los metadatos; el resto en ``final_metrics``.
DELTA_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "loss",
    "training_time",
)

# Base indicada para cada diferencia relativa (solo cuando el denominador es válido).
RELATIVE_BASIS = "fraction of baseline value: (candidate - baseline) / baseline"

COMPARISON_TYPES: tuple[str, ...] = (
    "identical",
    "controlled",
    "multi_variable",
    "non_comparable",
)

# Métricas de validación (mejor época), derivadas de ``validation_metrics`` en los
# metadatos de la ejecución. Se usan para la comparación exploratoria entre semillas
# (no para la evaluación final, que se reserva al conjunto de prueba).
_VALIDATION_METRICS: tuple[str, ...] = ("val_loss", "val_accuracy")

# Alias público para consumidores (p. ej. :mod:`prebenchmark.aggregate`).
VALIDATION_METRICS = _VALIDATION_METRICS

_SPLIT_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Documented repository default baseline
# ---------------------------------------------------------------------------


def documented_baseline_config() -> ExperimentConfig:
    """La configuración de referencia documentada.

    ReLU + Adam + lr 0.001 + He + CrossEntropyLoss. Se expone para que los
    llamadores puedan optar por ella de forma explícita; nunca se asume de
    forma implícita. El dataset y la arquitectura concretos se toman de las
    ejecuciones al seleccionar la línea base.
    """
    return ExperimentConfig(
        activation=BASELINE_ACTIVATION,
        optimizer=BASELINE_OPTIMIZER,
        lr=BASELINE_LR,
        init=BASELINE_INIT,
        loss="cross_entropy",
    )


def _matches_documented_default(meta: Mapping[str, Any]) -> bool:
    """Verdadero cuando la ejecución usa las cuatro dimensiones de la línea base.

    Se compara por dimensiones (no por ``config_id``) para que la selección
    funcione con cualquier dataset, arquitectura o protocolo de convergencia.
    """
    expected: tuple[tuple[str, Any], ...] = (
        ("activation", BASELINE_ACTIVATION),
        ("optimizer", BASELINE_OPTIMIZER),
        ("lr", BASELINE_LR),
        ("init", BASELINE_INIT),
    )
    for key, expected_value in expected:
        value = _field(meta, key)
        if key == "lr":
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
        if value != expected_value:
            return False
    return True


# ---------------------------------------------------------------------------
# Normalización de metadatos (acepta registros del ejecutor y registros de prueba)
# ---------------------------------------------------------------------------


def _state(meta: Mapping[str, Any]) -> str | None:
    return meta.get("execution_state") or meta.get("state")


def _config(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    return meta.get("config") or meta.get("config_snapshot") or {}


def _field(meta: Mapping[str, Any], key: str) -> Any:
    """Valor de campo de nivel superior; si no existe, recurre a la configuración
    incrustada en la instantánea (config snapshot).
    """
    value = meta.get(key)
    return value if value is not None else _config(meta).get(key)


def _dataset(meta: Mapping[str, Any]) -> tuple[Any, Any]:
    ds = meta.get("dataset") or _config(meta).get("dataset") or {}
    if isinstance(ds, Mapping):
        return (ds.get("name"), ds.get("version"))
    return (ds, None)  # plain-string fallback


def _architecture(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    arch = meta.get("architecture") or _config(meta).get("architecture") or {}
    if isinstance(arch, Mapping):
        return (
            arch.get("name"),
            arch.get("input_dim"),
            tuple(arch.get("hidden_layers") or ()),
            arch.get("output_dim"),
        )
    return (arch,)


def _split(meta: Mapping[str, Any]) -> tuple[float, ...]:
    split = meta.get("split") or _config(meta).get("split") or {}
    if isinstance(split, Mapping):
        return (
            float(split.get("train", 0.6)),
            float(split.get("val", 0.2)),
            float(split.get("test", 0.2)),
            bool(split.get("stratify", True)),
        )
    return ()


def _class_weight(meta: Mapping[str, Any]) -> str | None:
    """Valor de ``preprocessing.class_weight`` de la ejecución.

    Vive anidado en la sección ``preprocessing`` de la instantánea de configuración,
    por lo que no se puede leer con :func:`_field`.
    """
    preprocessing = meta.get("preprocessing") or _config(meta).get("preprocessing") or {}
    if isinstance(preprocessing, Mapping):
        value = preprocessing.get("class_weight")
        return None if value is None else str(value)
    return None


def _convergence(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    conv = meta.get("convergence") or _config(meta).get("convergence") or {}
    if isinstance(conv, Mapping):
        return (
            conv.get("enabled"),
            conv.get("metric"),
            conv.get("patience"),
            conv.get("min_delta"),
            conv.get("mode"),
        )
    return ()


def _final_metrics(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    return meta.get("final_metrics") or {}


def _eval_set(meta: Mapping[str, Any]) -> str | None:
    metrics = _final_metrics(meta)
    return metrics.get("evaluation_set") or metrics.get("eval_set")


def _failure_reason(meta: Mapping[str, Any]) -> str | None:
    failure = meta.get("failure")
    if isinstance(failure, Mapping) and failure.get("reason"):
        return str(failure["reason"])
    return str(failure) if failure else None


def _metric_value(meta: Mapping[str, Any], metric: str) -> float | None:
    if metric == "training_time":
        value = _field(meta, "training_time")
    elif metric in _VALIDATION_METRICS:
        validation = meta.get("validation_metrics") or {}
        key = "loss" if metric == "val_loss" else "accuracy"
        value = validation.get(key)
    else:
        value = _final_metrics(meta).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _splits_close(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    return all(abs(x - y) <= _SPLIT_TOLERANCE for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Public normalization surface
# ---------------------------------------------------------------------------
#
# El ejecutor escribe el estado canónico ``execution_state`` para cada resultado
# (COMPLETED / COMPLETED_WITH_WARNINGS / FAILED) e incrusta la configuración en
# ``config_snapshot``; los registros antiguos o de prueba pueden usar ``state`` y
# ``config``. Estos alias exponen una única normalización canónica para que los
# consumidores posteriores (aggregate.py y la capa de exportación de PR6) lean
# los registros *de forma idéntica*, sin acceder directamente a las claves
# ``state``/``execution_state``/``config``.

run_state = _state
run_config = _config
run_field = _field
run_dataset = _dataset
run_architecture = _architecture
run_metric_value = _metric_value
run_eval_set = _eval_set
run_failure_reason = _failure_reason
run_class_weight = _class_weight


# ---------------------------------------------------------------------------
# Filtros de comparación
# ---------------------------------------------------------------------------


def filter_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    dataset: Any = None,
    architecture: Any = None,
    activation: Any = None,
    optimizer: Any = None,
    lr: Any = None,
    init: Any = None,
    batch_size: Any = None,
    seed: Any = None,
    run_type: Any = None,
    status: Any = None,
    state: Any = None,
) -> list[dict[str, Any]]:
    """Mantiene solo las ejecuciones cuyos campos normalizados coinciden con todos
    los filtros proporcionados.

    ``status`` es equivalente a ``state`` (estado de ejecución): se aceptan
    ambas ortografías, pero no ambas a la vez. Filtrar las ejecuciones *antes*
    de compararlas permite aplicar el escenario "filtrar a un conjunto
    comparable" especificado.
    """
    if status is not None and state is not None:
        raise ValueError("pass either 'status' or 'state', not both")
    status = status if state is None else state

    matched: list[dict[str, Any]] = []
    for meta in runs:
        if dataset is not None and _dataset(meta)[0] != dataset:
            continue
        if architecture is not None and _architecture(meta)[0] != architecture:
            continue
        if activation is not None and _field(meta, "activation") != activation:
            continue
        if optimizer is not None and _field(meta, "optimizer") != optimizer:
            continue
        if lr is not None and _field(meta, "lr") != lr:
            continue
        if init is not None and _field(meta, "init") != init:
            continue
        if batch_size is not None and _field(meta, "batch_size") != batch_size:
            continue
        if seed is not None and _field(meta, "seed") != seed:
            continue
        if run_type is not None and _field(meta, "run_type") != run_type:
            continue
        if status is not None and _state(meta) != status:
            continue
        matched.append(dict(meta))
    return matched


# ---------------------------------------------------------------------------
# Selección de la línea base (explícita, registrada e inmutable por contexto de comparación)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineSelection:
    """Instantánea de la línea base seleccionada.

    Inmutable intencionalmente: la línea base DEBE permanecer estable para un
    contexto de comparación determinado; seleccionar una línea base distinta
    crea una nueva selección (junto con una nueva justificación y una nueva
    marca de tiempo).
    """

    run_id: str
    config_id: str
    state: str
    timestamp: str
    rationale: str
    metadata: Mapping[str, Any]


def select_baseline(
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline_run_id: str | None = None,
    baseline_config_id: str | None = None,
    use_documented_default: bool = False,
    rationale: str | None = None,
    timestamp: str | None = None,
) -> BaselineSelection:
    """Selecciona una ejecución como línea base -- nunca de forma implícita.

    Se requiere exactamente uno de los siguientes: ``baseline_run_id`` o
    ``baseline_config_id``, o bien ``use_documented_default=True`` para optar
    por la configuración predeterminada documentada (ReLU + Adam + lr 0.001 +
    He + CrossEntropy), seleccionada por sus cuatro dimensiones libres para que
    funcione con cualquier dataset, arquitectura o protocolo de convergencia. En
    caso contrario, se genera un ``ValueError``. Cuando varias ejecuciones
    comparten el mismo ``baseline_config_id``, se selecciona la más temprana
    (según ``start_time``) y la selección registra su justificación y una marca
    de tiempo UTC.
    """
    if baseline_run_id is not None and baseline_config_id is not None:
        raise ValueError("pass exactly one of baseline_run_id / baseline_config_id")
    if baseline_run_id is None and baseline_config_id is None and not use_documented_default:
        raise ValueError(
            "baseline must be selected explicitly: pass baseline_run_id, "
            "baseline_config_id, or use_documented_default=True"
        )

    if baseline_run_id is not None:
        matches = [m for m in runs if m.get("run_id") == baseline_run_id]
        if not matches:
            raise ValueError(f"baseline run '{baseline_run_id}' not found")
        chosen = matches[0]
    elif baseline_config_id is not None:
        matches = [m for m in runs if m.get("config_id") == baseline_config_id]
        if not matches:
            raise ValueError(f"no run found for baseline config_id '{baseline_config_id}'")
        chosen = sorted(
            matches,
            key=lambda m: (m.get("start_time") or "", m.get("run_id") or ""),
        )[0]
    else:
        matches = [m for m in runs if _matches_documented_default(m)]
        if not matches:
            raise ValueError(
                "no run found matching the documented default baseline "
                "(ReLU + Adam + lr 0.001 + He)"
            )
        chosen = sorted(
            matches,
            key=lambda m: (m.get("start_time") or "", m.get("run_id") or ""),
        )[0]

    state = _state(chosen) or "UNKNOWN"
    if state == "FAILED":
        raise ValueError(
            f"baseline run '{chosen.get('run_id')}' is FAILED; cannot compare against it"
        )

    if rationale is None:
        if baseline_run_id is not None:
            rationale = f"explicitly selected baseline run {chosen.get('run_id')}"
        elif use_documented_default:
            dataset_name = (_config(chosen).get("dataset") or {}).get("name") or "unknown"
            rationale = (
                "documented default baseline: ReLU + Adam + lr 0.001 + He "
                f"(matched by the four free dimensions; dataset: {dataset_name})"
            )
        else:
            rationale = f"explicitly selected earliest run matching config_id {baseline_config_id}"
    if not str(rationale).strip():
        raise ValueError("baseline rationale must be a non-empty string")

    return BaselineSelection(
        run_id=str(chosen.get("run_id") or "?"),
        config_id=str(chosen.get("config_id") or "?"),
        state=state,
        timestamp=timestamp or utc_timestamp(),
        rationale=str(rationale),
        metadata=chosen,
    )


# ---------------------------------------------------------------------------
# Guardas de comparabilidad
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparabilityResult:
    """Resultado de la comprobación de comparabilidad entre dos ejecuciones."""

    comparable: bool
    reasons: tuple[str, ...] = ()
    evaluation_set: str | None = None


def check_comparability(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> ComparabilityResult:
    """Indica si las métricas del ``candidate`` pueden compararse con las de la
    ``baseline``.

    Guardas: dataset (nombre + versión), arquitectura (nombre, input_dim,
    hidden_layers, output_dim), proporciones y estratificación de la partición,
    max_epochs, configuración de convergencia, run_type, conjunto de evaluación
    y estado de ejecución. Cualquier discrepancia -> ``comparable=False`` con
    motivos legibles; los llamadores deben tratar las diferencias como N/A.
    """
    reasons: list[str] = []

    base_set, cand_set = _eval_set(baseline), _eval_set(candidate)
    if base_set is None or cand_set is None:
        reasons.append("evaluation set is not labelled on at least one run")
    elif base_set != cand_set:
        reasons.append(
            f"evaluation sets differ: baseline on '{base_set}', candidate on '{cand_set}'"
        )

    if _dataset(baseline) != _dataset(candidate):
        reasons.append("datasets differ (name or version)")
    if _architecture(baseline) != _architecture(candidate):
        reasons.append("architectures differ (name, input_dim, hidden_layers or output_dim)")
    if not _splits_close(_split(baseline), _split(candidate)):
        reasons.append("split proportions or stratification differ")
    if _field(baseline, "max_epochs") != _field(candidate, "max_epochs"):
        reasons.append("max_epochs differ (training protocol)")
    if _convergence(baseline) != _convergence(candidate):
        reasons.append("convergence settings differ (training protocol)")
    if _field(baseline, "run_type") != _field(candidate, "run_type"):
        reasons.append("run_type differs (prebenchmark vs benchmark)")

    base_state, cand_state = _state(baseline), _state(candidate)
    if base_state == "FAILED":
        reasons.append("baseline run is FAILED")
    if cand_state == "FAILED":
        failure = _failure_reason(candidate)
        reasons.append(f"candidate run is FAILED{f': {failure}' if failure else ''}")

    return ComparabilityResult(
        comparable=not reasons,
        reasons=tuple(reasons),
        evaluation_set=cand_set,
    )


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------


def _absolute_delta(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    if not (math.isfinite(base) and math.isfinite(cand)):
        return None
    return cand - base


def _relative_delta(base: float | None, cand: float | None) -> float | None:
    """Diferencia relativa solo si el denominador es interpretable: distinto de
    cero y finito.
    """
    if base is None or cand is None or base == 0:
        return None
    if not (math.isfinite(base) and math.isfinite(cand)):
        return None
    return (cand - base) / base


# ---------------------------------------------------------------------------
# Tabla de comparación
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """Una ejecución candidata comparada con la línea base."""

    run_id: str
    config_id: str
    state: str
    evaluation_set: str | None
    comparison_type: str  # identical | controlled | multi_variable | non_comparable
    varied_dimensions: tuple[str, ...]
    informative_only: bool
    comparable: bool
    comparability_reasons: tuple[str, ...]
    failure_reason: str | None
    metrics: Mapping[str, float]  # raw candidate metrics (non-None values)
    deltas: Mapping[str, float | None]  # absolute deltas; None = N/A
    relative_deltas: Mapping[str, float | None]
    relative_basis: str


@dataclass(frozen=True)
class ComparisonTable:
    """Salida estructurada de la comparación: perfil completo, diferencias,
    indicadores y advertencias.

    Las filas conservan el orden de entrada de los candidatos; no se aplica
    ningún orden (no existe una puntuación compuesta para ordenar).
    """

    baseline_run_id: str
    baseline_config_id: str
    baseline_state: str
    baseline_rationale: str
    baseline_timestamp: str
    evaluation_set: str | None
    baseline_metrics: Mapping[str, float]
    rows: tuple[ComparisonRow, ...]

    def to_records(self, *, include_relative: bool = True) -> list[dict[str, Any]]:
        """Convierte la tabla en diccionarios de fila preparados para CSV."""
        records: list[dict[str, Any]] = []
        for row in self.rows:
            record: dict[str, Any] = {
                "run_id": row.run_id,
                "config_id": row.config_id,
                "state": row.state,
                "eval_set": row.evaluation_set,
                "comparability_flag": "comparable" if row.comparable else "non_comparable",
                "comparison_type": row.comparison_type,
                "informative_only": row.informative_only,
                "controlled_variable": (
                    row.varied_dimensions[0] if len(row.varied_dimensions) == 1 else ""
                ),
                "varied_dimensions": "|".join(row.varied_dimensions),
                "warnings": "|".join(row.comparability_reasons),
                "failure_reason": row.failure_reason or "",
            }
            for metric, value in row.metrics.items():
                record[metric] = value
            for metric in DELTA_METRICS:
                record[f"delta_{metric}"] = row.deltas.get(metric)
                if include_relative:
                    record[f"rel_{metric}"] = row.relative_deltas.get(metric)
            records.append(record)
        return records


def _build_row(
    baseline_meta: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    include_relative: bool,
) -> ComparisonRow:
    comparability = check_comparability(baseline_meta, candidate)
    state = _state(candidate) or "UNKNOWN"
    failure = _failure_reason(candidate)

    varied = tuple(
        dim for dim in SPACE_DIMENSIONS if _field(baseline_meta, dim) != _field(candidate, dim)
    )
    if _class_weight(baseline_meta) != _class_weight(candidate):
        varied = varied + ("class_weight",)
    if not comparability.comparable:
        comparison_type = "non_comparable"
    elif not varied:
        comparison_type = "identical"
    elif len(varied) == 1:
        comparison_type = "controlled"
    else:
        comparison_type = "multi_variable"

    metrics: dict[str, float] = {}
    for metric in DELTA_METRICS:
        value = _metric_value(candidate, metric)
        if value is not None:
            metrics[metric] = value

    if comparability.comparable:
        deltas = {
            metric: _absolute_delta(
                _metric_value(baseline_meta, metric), _metric_value(candidate, metric)
            )
            for metric in DELTA_METRICS
        }
        if include_relative:
            relative_deltas = {
                metric: _relative_delta(
                    _metric_value(baseline_meta, metric), _metric_value(candidate, metric)
                )
                for metric in DELTA_METRICS
            }
        else:
            relative_deltas = dict.fromkeys(DELTA_METRICS)
    else:
        deltas = dict.fromkeys(DELTA_METRICS)
        relative_deltas = dict.fromkeys(DELTA_METRICS)

    return ComparisonRow(
        run_id=str(candidate.get("run_id") or "?"),
        config_id=str(candidate.get("config_id") or "?"),
        state=state,
        evaluation_set=comparability.evaluation_set,
        comparison_type=comparison_type,
        varied_dimensions=varied,
        informative_only=comparison_type in ("multi_variable", "non_comparable"),
        comparable=comparability.comparable,
        comparability_reasons=comparability.reasons,
        failure_reason=failure,
        metrics=metrics,
        deltas=deltas,
        relative_deltas=relative_deltas,
        relative_basis=RELATIVE_BASIS if include_relative else "",
    )


def compare_to_baseline(
    baseline: BaselineSelection | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    include_relative: bool = True,
) -> ComparisonTable:
    """Compara cada ejecución ``candidate`` con una línea base inmutable.

    ``baseline`` puede ser una :class:`BaselineSelection` (de
    :func:`select_baseline`) o un mapa de metadatos sin procesar (envuelto con
    una justificación provisional). Si la línea base está fallida, se rechaza
    toda la comparación; las ejecuciones candidatas fallidas se enumeran con su
    motivo y no aportan diferencias.
    """
    if not isinstance(baseline, BaselineSelection):
        baseline = BaselineSelection(
            run_id=str(baseline.get("run_id") or "?"),
            config_id=str(baseline.get("config_id") or "?"),
            state=_state(baseline) or "UNKNOWN",
            timestamp="",
            rationale="baseline supplied as raw metadata mapping (no recorded rationale)",
            metadata=baseline,
        )
    if baseline.state == "FAILED":
        raise ValueError(
            f"baseline run '{baseline.run_id}' is FAILED; cannot compute deltas against it"
        )

    baseline_metrics: dict[str, float] = {}
    for metric in DELTA_METRICS:
        value = _metric_value(baseline.metadata, metric)
        if value is not None:
            baseline_metrics[metric] = value

    rows = tuple(
        _build_row(baseline.metadata, candidate, include_relative=include_relative)
        for candidate in candidates
    )

    return ComparisonTable(
        baseline_run_id=baseline.run_id,
        baseline_config_id=baseline.config_id,
        baseline_state=baseline.state,
        baseline_rationale=baseline.rationale,
        baseline_timestamp=baseline.timestamp,
        evaluation_set=_eval_set(baseline.metadata),
        baseline_metrics=baseline_metrics,
        rows=rows,
    )
