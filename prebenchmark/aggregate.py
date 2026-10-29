"""Agrupación por repeticiones y estadísticas agregadas por grupo.

Funciones puras sobre registros de metadatos de ejecución (tal como los devuelve
:meth:`prebenchmark.store.ResultsStore.query_runs`): sin E/S, sin torch, por lo
que el módulo puede importarse de forma segura desde cualquier sitio.

Semántica (decisiones vinculantes para este fragmento):

* **Identidad de agrupación** -- ``config_id`` es un hash del contenido de la
  *configuración completa*, **incluida la ``semilla``** (``seed``)
  (:mod:`prebenchmark.identity`). Las ejecuciones del mismo protocolo con
  distintas semillas tienen, por tanto, valores distintos de ``config_id``, pero
  la especificación exige que compartan un grupo de repeticiones. El modo
  predeterminado de :func:`group_runs` (``grouping="config"``) agrupa por una
  identidad de configuración **normalizada por semilla** (se elimina ``seed``),
  de modo que las repeticiones con múltiples semillas se colapsan en un solo
  grupo, mientras que cualquier diferencia de protocolo (dataset, arquitectura,
  partición, max_epochs, convergencia, class_weight, ...) sigue separándolos.
  ``grouping="config_id"`` mantiene el comportamiento estricto sensible a la
  semilla cuando se solicita explícitamente.
* **Estadísticas** -- por métrica (exactitud, exactitud balanceada,
  precisión/exhaustividad/F1 macro, pérdida, tiempo de entrenamiento, más
  ``val_loss``/``val_accuracy`` de la mejor época): ``n``,
  ``mean``, ``std`` muestral (``ddof=1``, ``None`` si ``n < 2``), ``min``,
  ``max`` y ``range`` (``max - min``). Solo contribuyen las ejecuciones no
  fallidas con un valor finito para esa métrica.
* **Ejecuciones fallidas** -- se excluyen de las estadísticas, pero nunca se
  omiten silenciosamente: siempre se reportan ``n_total``, ``n_valid``,
  ``n_failed``, ``n_unstable`` y sus proporciones, y cada miembro se enumera
  con su estado y motivo.
* **Ejecuciones inestables** -- se cuentan cuando están clasificadas, ya sea
  por un estado técnico ``UNSTABLE`` de ejecución o por una etiqueta
  exploratoria ``UNSTABLE`` en el registro. Estas ejecuciones siguen
  contribuyendo a las estadísticas (la especificación indica que la inestabilidad
  no es un fallo); simplemente se divulgan por separado.
* **Repeticiones insuficientes** -- cuando ``n_valid < min_repetitions``
  (valor predeterminado **3**, configurable y registrado), el grupo se marca;
  los valores siguen siendo visibles, pero nunca se presentan como fiables.
* **Contexto del espacio** -- el contexto del espacio de 36 dimensiones se
  transporta en cada grupo mediante ``space_key``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .compare import (
    DELTA_METRICS,
    VALIDATION_METRICS,
    run_config,
    run_dataset,
    run_eval_set,
    run_failure_reason,
    run_field,
    run_metric_value,
    run_state,
)
from .decisions import EXPLORATORY_LABELS, TECHNICAL_STATES
from .identity import utc_timestamp

__all__ = [
    "AGGREGATE_METRICS",
    "DEFAULT_MIN_REPETITIONS",
    "EXPLORATORY_LABELS",
    "GROUP_KEY_PREFIX",
    "GROUPING_MODES",
    "TECHNICAL_STATES",
    "MetricStatistics",
    "RepetitionGroup",
    "RunGroup",
    "RunMember",
    "aggregate_group",
    "aggregate_records",
    "aggregate_runs",
    "compute_group_key",
    "group_runs",
]

# Métricas agregadas por grupo. Incluye el perfil de diferencias del comparador
# (métricas de prueba, para la evaluación final) más las métricas de validación de
# la mejor época, que son las que guían la comparación exploratoria entre semillas
# sin consumir el conjunto de prueba.
AGGREGATE_METRICS: tuple[str, ...] = DELTA_METRICS + VALIDATION_METRICS

# Número mínimo documentado de repeticiones válidas. Tamaño muestral mínimo
# para reportar una desviación estándar muestral sin implicar fiabilidad
# estadística.
DEFAULT_MIN_REPETITIONS = 3

GROUP_KEY_PREFIX = "grp-v1-"

# Identidades de agrupación admitidas: configuración normalizada por semilla
# (predeterminado, múltiples semillas) o ``config_id`` estricto y sensible a la
# semilla.
GROUPING_MODES: tuple[str, ...] = ("config", "config_id")

_UNSTABLE = "UNSTABLE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _exploratory_label(meta: Mapping[str, Any]) -> str | None:
    label = meta.get("exploratory_label") or meta.get("label")
    return str(label) if label else None


def _member_state(meta: Mapping[str, Any]) -> str | None:
    return run_state(meta)


def _is_failed(meta: Mapping[str, Any]) -> bool:
    return _member_state(meta) == "FAILED"


def _is_unstable(meta: Mapping[str, Any]) -> bool:
    return _member_state(meta) == _UNSTABLE or _exploratory_label(meta) == _UNSTABLE


def _config_id_of(meta: Mapping[str, Any]) -> str | None:
    value = meta.get("config_id")
    return str(value) if value else None


def _space_key_of(meta: Mapping[str, Any]) -> str | None:
    """Devuelve la clave del espacio (space key) del registro, reconstruyéndola
    a partir de la instantánea.

    Los metadatos del ejecutor no persisten ``space_key``; los registros del
    almacén sí lo hacen. Cuando no está presente, la instantánea se reconstruye
    como un ``ExperimentConfig`` y se aplica el hash con la función canónica
    :func:`prebenchmark.space.compute_space_key` (importada de forma diferida
    para mantener el peso de importación reducido).
    """
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
    except Exception:  # pragma: no cover - defensive: never crash aggregation
        return None


def _run_eval_sets(meta: Mapping[str, Any]) -> tuple[str, ...]:
    """Conjuntos de evaluación presentes en un registro de ejecución.

    Incluye el conjunto final (``final_metrics.evaluation_set``, normalmente
    ``test``) y el conjunto de la mejor época (``validation_metrics.evaluation_set``,
    normalmente ``validation``), de modo que la divulgación de una agregación que
    mezcla ambas fuentes sea honesta.
    """
    sets: list[str] = []
    final_set = run_eval_set(meta)
    if final_set:
        sets.append(str(final_set))
    validation = meta.get("validation_metrics")
    if isinstance(validation, Mapping):
        val_set = validation.get("evaluation_set")
        if val_set:
            sets.append(str(val_set))
    return tuple(sets)


def compute_group_key(
    meta: Mapping[str, Any],
    *,
    grouping: str = "config",
) -> str:
    """Clave de agrupación estable para ``meta``.

    ``grouping="config"`` (predeterminado) aplica hash a la configuración
    normalizada por semilla, de modo que distintas semillas del mismo protocolo
    comparten una misma clave. ``grouping="config_id"`` devuelve el
    ``config_id`` del registro tal cual (sensible a la semilla). Si no existe
    una instantánea de configuración, el único recurso disponible es el
    ``config_id`` del registro.
    """
    if grouping not in GROUPING_MODES:
        raise ValueError(f"grouping must be one of {GROUPING_MODES}, got {grouping!r}")

    config_id = _config_id_of(meta)
    if grouping == "config_id":
        return config_id or "grp-v1-unknown"

    config = run_config(meta)
    if isinstance(config, Mapping) and config:
        normalized = {key: value for key, value in config.items() if key != "seed"}
        payload = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{GROUP_KEY_PREFIX}{digest}"
    return config_id or "grp-v1-unknown"


# ---------------------------------------------------------------------------
# Agrupación
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunGroup:
    """Conjunto de ejecuciones que comparten una misma identidad de
    configuración/protocolo.

    Solo agrupación -- las estadísticas se encuentran en
    :class:`RepetitionGroup`.
    """

    group_key: str
    config_id: str
    config_ids: tuple[str, ...]
    space_key: str | None
    config_snapshot: Mapping[str, Any]
    runs: tuple[Mapping[str, Any], ...]
    created_at: str

    @property
    def size(self) -> int:
        return len(self.runs)


def group_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    grouping: str = "config",
    created_at: str | None = None,
) -> list[RunGroup]:
    """Agrupa las ejecuciones por identidad de configuración, en orden
    determinista.

    El valor predeterminado ``grouping="config"`` colapsa distintas semillas
    del mismo protocolo en un único grupo (especificación: soporte para
    múltiples semillas). Use ``grouping="config_id"`` para una agrupación
    estricta sensible a la semilla.
    """
    timestamp = created_at or utc_timestamp()
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for meta in runs:
        buckets.setdefault(compute_group_key(meta, grouping=grouping), []).append(meta)

    groups: list[RunGroup] = []
    for key in sorted(buckets):
        members = sorted(
            buckets[key],
            key=lambda m: (m.get("start_time") or "", m.get("run_id") or ""),
        )
        config_ids = tuple(sorted({cid for cid in (_config_id_of(m) for m in members) if cid}))
        groups.append(
            RunGroup(
                group_key=key,
                config_id=config_ids[0] if config_ids else "?",
                config_ids=config_ids,
                space_key=_space_key_of(members[0]),
                config_snapshot=dict(run_config(members[0]) or {}),
                runs=tuple(members),
                created_at=timestamp,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricStatistics:
    """Estadísticas agregadas para una única métrica sobre las ejecuciones
    válidas de un grupo.
    """

    metric: str
    n: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    spread: float | None
    sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "range": self.spread,
            "sufficient": self.sufficient,
        }


def _compute_statistics(
    values: Sequence[float], metric: str, min_repetitions: int
) -> MetricStatistics:
    finite = [float(value) for value in values if _is_finite_number(value)]
    n = len(finite)
    if n == 0:
        return MetricStatistics(metric, 0, None, None, None, None, None, False)

    mean = sum(finite) / n
    if n >= 2:  # desviación estándar muestral (ddof=1); no definida si n < 2
        variance = sum((value - mean) ** 2 for value in finite) / (n - 1)
        std: float | None = math.sqrt(variance)
    else:
        std = None
    minimum = min(finite)
    maximum = max(finite)
    return MetricStatistics(
        metric=metric,
        n=n,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        spread=maximum - minimum,
        sufficient=n >= min_repetitions,
    )


@dataclass(frozen=True)
class RunMember:
    """Un miembro de un grupo de repeticiones, con su decisión de inclusión."""

    run_id: str
    config_id: str | None
    seed: int | None
    state: str | None
    included: bool
    metric_count: int
    failure_reason: str | None
    warnings: tuple[str, ...]
    exploratory_label: str | None


@dataclass(frozen=True)
class RepetitionGroup:
    """Vista agregada de un grupo de repeticiones."""

    group_key: str
    config_id: str
    config_ids: tuple[str, ...]
    space_key: str | None
    config_snapshot: Mapping[str, Any]
    created_at: str
    metrics: Mapping[str, MetricStatistics]
    members: tuple[RunMember, ...]
    seeds: tuple[int, ...]
    evaluation_sets: tuple[str, ...]
    n_total: int
    n_valid: int
    n_failed: int
    n_unstable: int
    failed_proportion: float
    unstable_proportion: float
    min_repetitions: int
    insufficient_repetitions: bool
    state_counts: Mapping[str, int]

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self.metrics)

    def to_record(self) -> dict[str, Any]:
        """Convierte el grupo y sus estadísticas en un diccionario de fila
        preparado para CSV.
        """
        config = self.config_snapshot or {}
        dataset = config.get("dataset")
        record: dict[str, Any] = {
            "group_key": self.group_key,
            "config_id": self.config_id,
            "config_ids": "|".join(self.config_ids),
            "space_key": self.space_key or "",
            "activation": config.get("activation"),
            "optimizer": config.get("optimizer"),
            "lr": config.get("lr"),
            "init": config.get("init"),
            "dataset": dataset.get("name") if isinstance(dataset, Mapping) else dataset,
            "batch_size": config.get("batch_size"),
            "seeds": "|".join(str(seed) for seed in self.seeds),
            "run_ids": "|".join(member.run_id for member in self.members),
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_failed": self.n_failed,
            "n_unstable": self.n_unstable,
            "failed_proportion": self.failed_proportion,
            "unstable_proportion": self.unstable_proportion,
            "min_repetitions": self.min_repetitions,
            "insufficient_repetitions": self.insufficient_repetitions,
            "evaluation_sets": "|".join(self.evaluation_sets),
            "created_at": self.created_at,
        }
        for metric in AGGREGATE_METRICS:
            stats = self.metrics.get(metric)
            record[f"{metric}_n"] = stats.n if stats else 0
            record[f"{metric}_mean"] = stats.mean if stats else None
            record[f"{metric}_std"] = stats.std if stats else None
            record[f"{metric}_min"] = stats.minimum if stats else None
            record[f"{metric}_max"] = stats.maximum if stats else None
            record[f"{metric}_range"] = stats.spread if stats else None
        return record


def _member_of(meta: Mapping[str, Any]) -> RunMember:
    seed = run_field(meta, "seed")
    warnings = meta.get("warnings") or ()
    metric_count = sum(
        1 for metric in AGGREGATE_METRICS if _is_finite_number(run_metric_value(meta, metric))
    )
    failed = _is_failed(meta)
    return RunMember(
        run_id=str(meta.get("run_id") or "?"),
        config_id=_config_id_of(meta),
        seed=int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None,
        state=_member_state(meta),
        included=not failed,
        metric_count=metric_count,
        failure_reason=run_failure_reason(meta),
        warnings=tuple(str(w) for w in warnings),
        exploratory_label=_exploratory_label(meta),
    )


def aggregate_group(
    group: RunGroup,
    *,
    min_repetitions: int = DEFAULT_MIN_REPETITIONS,
) -> RepetitionGroup:
    """Calcula las estadísticas agregadas para un :class:`RunGroup`.

    ``min_repetitions`` se registra en el resultado (valor predeterminado
    :data:`DEFAULT_MIN_REPETITIONS` = 3) y determina la marca de repeticiones
    insuficientes; nunca es una regla oculta e inamovible.
    """
    if min_repetitions < 1:
        raise ValueError("min_repetitions must be >= 1")
    if group.size == 0:
        raise ValueError("cannot aggregate an empty run group")

    members = tuple(_member_of(meta) for meta in group.runs)
    member_metas = group.runs

    valid_metas = [meta for meta in member_metas if not _is_failed(meta)]
    n_total = len(member_metas)
    n_valid = len(valid_metas)
    n_failed = n_total - n_valid
    n_unstable = sum(1 for meta in member_metas if _is_unstable(meta))

    metrics = {
        metric: _compute_statistics(
            [value for meta in valid_metas if (value := run_metric_value(meta, metric)) is not None],
            metric,
            min_repetitions,
        )
        for metric in AGGREGATE_METRICS
    }

    seeds = tuple(
        sorted(
            {
                member.seed
                for member in members
                if member.seed is not None
            }
        )
    )
    evaluation_sets = tuple(
        sorted(
            {
                eval_set
                for meta in member_metas
                for eval_set in (_run_eval_sets(meta))
            }
        )
    )
    state_counts: dict[str, int] = {}
    for meta in member_metas:
        state = _member_state(meta) or "UNKNOWN"
        state_counts[state] = state_counts.get(state, 0) + 1

    return RepetitionGroup(
        group_key=group.group_key,
        config_id=group.config_id,
        config_ids=group.config_ids,
        space_key=group.space_key,
        config_snapshot=group.config_snapshot,
        created_at=group.created_at,
        metrics=metrics,
        members=members,
        seeds=seeds,
        evaluation_sets=evaluation_sets,
        n_total=n_total,
        n_valid=n_valid,
        n_failed=n_failed,
        n_unstable=n_unstable,
        failed_proportion=(n_failed / n_total if n_total else 0.0),
        unstable_proportion=(n_unstable / n_total if n_total else 0.0),
        min_repetitions=min_repetitions,
        insufficient_repetitions=n_valid < min_repetitions,
        state_counts=state_counts,
    )


def aggregate_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    grouping: str = "config",
    min_repetitions: int = DEFAULT_MIN_REPETITIONS,
    created_at: str | None = None,
) -> list[RepetitionGroup]:
    """Agrupa ``runs`` y agrega cada grupo resultante (función de conveniencia)."""
    return [
        aggregate_group(group, min_repetitions=min_repetitions)
        for group in group_runs(runs, grouping=grouping, created_at=created_at)
    ]


def aggregate_records(groups: Sequence[RepetitionGroup]) -> list[dict[str, Any]]:
    """Convierte los grupos agregados en diccionarios de fila preparados para CSV."""
    return [group.to_record() for group in groups]
