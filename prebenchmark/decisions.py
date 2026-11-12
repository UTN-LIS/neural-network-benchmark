"""Registro de decisiones manual: solo adición, justificado, vinculado a actor y
entidad.

El registro de decisiones es *exclusivamente manual*. No hay aquí ninguna
clasificación automática de ejecuciones: deliberadamente no existe un motor de
reglas, ningún umbral ni descartes automáticos. Un analista (humano o proceso
identificado) registra una decisión con una justificación no vacía, un actor,
una marca de tiempo y un vínculo a la entidad a la que hace referencia
(``run_id`` y/o ``config_id`` y/o ``group_id``). El registro se añade mediante
:meth:`prebenchmark.store.ResultsStore.append_decision`, de modo que las
entradas existentes nunca se modifican ni se eliminan y se puede recuperar el
historial completo.

Se mantienen deliberadamente independientes dos clasificaciones:

* **Estado técnico de ejecución** -- ``COMPLETED`` / ``COMPLETED_WITH_WARNINGS``
  / ``UNSTABLE`` / ``FAILED`` / ``INCOMPLETE``. Este estado reside en el
  registro de la ejecución (escrito por el ejecutor); el registro de decisiones
  nunca lo modifica.
* **Etiqueta exploratoria** -- ``VIABLE`` / ``INTERESTING`` / ``UNSTABLE`` /
  ``NON_VIABLE`` / ``PENDING_REVIEW``. Es un campo independiente del registro
  de decisiones y puede cambiar con el tiempo sin afectar al estado técnico de
  ejecución.

Se rechaza el paso de un estado técnico en un lugar donde se espera una
etiqueta exploratoria (``label=``), lo que constituye la protección concreta
que garantiza que la etiqueta sea independiente del estado técnico de ejecución.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .identity import new_decision_id, utc_timestamp
from .store import ResultsStore

__all__ = [
    "DECISION_TYPES",
    "EXPLORATORY_LABELS",
    "TECHNICAL_STATES",
    "decision_to_record",
    "decisions_to_records",
    "is_exploratory_label",
    "is_technical_state",
    "record_decision",
]

# ---------------------------------------------------------------------------
# Canonical classification vocabulary (documented defaults, no hidden values)
# ---------------------------------------------------------------------------

# Estados técnicos de ejecución (establecidos por el ejecutor en el registro de
# la ejecución).
TECHNICAL_STATES: tuple[str, ...] = (
    "COMPLETED",
    "COMPLETED_WITH_WARNINGS",
    "UNSTABLE",
    "FAILED",
    "INCOMPLETE",
)

# Etiquetas exploratorias de evaluación (asignadas manualmente, independientes
# del estado).
EXPLORATORY_LABELS: tuple[str, ...] = (
    "VIABLE",
    "INTERESTING",
    "UNSTABLE",
    "NON_VIABLE",
    "PENDING_REVIEW",
)

# Tipos de decisiones manuales documentados por los requisitos (keep / exclude /
# repeat / investigate) más los casos de validación y registro de etiquetas.
# Ampliar este conjunto es un cambio deliberado y revisable.
DECISION_TYPES: tuple[str, ...] = (
    "keep",
    "exclude",
    "repeat",
    "investigate",
    "validate",
    "label",
)


def is_technical_state(value: Any) -> bool:
    """Verdadero cuando ``value`` indica un estado técnico de ejecución."""
    return str(value) in TECHNICAL_STATES


def is_exploratory_label(value: Any) -> bool:
    """Verdadero cuando ``value`` indica una etiqueta exploratoria de evaluación."""
    return str(value) in EXPLORATORY_LABELS


# ---------------------------------------------------------------------------
# Registro (solo manual; solo adición mediante el almacén)
# ---------------------------------------------------------------------------


def record_decision(
    store: ResultsStore,
    experiment_id: str,
    *,
    decision_type: str,
    justification: str,
    actor: str,
    run_id: str | None = None,
    config_id: str | None = None,
    group_id: str | None = None,
    label: str | None = None,
    evidence: Any = None,
    timestamp: str | None = None,
    decision_id: str | None = None,
) -> dict[str, Any]:
    """Añade un registro de decisión manual a ``decisions.jsonl``.

    ``justification`` y ``actor`` no deben estar vacíos, se requiere al menos
    una referencia a una entidad y ``decision_type`` debe ser un tipo
    documentado. ``label`` (cuando se proporciona) debe ser una etiqueta
    exploratoria -- se rechaza un estado técnico, lo que mantiene independientes
    las dos clasificaciones. Nunca se infiere ningún valor: pase ``label=None``
    para registrar una decisión que no lleve etiqueta exploratoria.
    """
    if not str(justification).strip():
        raise ValueError("decision justification must be a non-empty string")
    if not str(actor).strip():
        raise ValueError("decision actor must be a non-empty string")
    if not any(value for value in (run_id, config_id, group_id)):
        raise ValueError("a decision must reference a run_id, config_id or group_id")
    if decision_type not in DECISION_TYPES:
        raise ValueError(
            f"unknown decision_type {decision_type!r}; expected one of {DECISION_TYPES}"
        )
    if label is not None and not is_exploratory_label(label):
        if is_technical_state(label):
            raise ValueError(
                f"{label!r} is a technical execution state, not an exploratory "
                f"label; expected one of {EXPLORATORY_LABELS}"
            )
        raise ValueError(
            f"unknown exploratory label {label!r}; expected one of {EXPLORATORY_LABELS}"
        )

    return store.append_decision(
        experiment_id,
        decision_type=decision_type,
        justification=justification,
        actor=actor,
        label=label,
        run_id=run_id,
        config_id=config_id,
        group_id=group_id,
        evidence=evidence,
        timestamp=timestamp or utc_timestamp(),
        decision_id=decision_id or new_decision_id(),
    )


# ---------------------------------------------------------------------------
# Funciones auxiliares de exportación (estructuradas / preparadas para CSV)
# ---------------------------------------------------------------------------


def decision_to_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convierte un registro de decisión (incluida la entidad) en un diccionario
    preparado para CSV.
    """
    entity = record.get("entity") or {}
    return {
        "decision_id": record.get("decision_id"),
        "run_id": entity.get("run_id", ""),
        "config_id": entity.get("config_id", ""),
        "group_id": entity.get("group_id", ""),
        "decision_type": record.get("decision_type"),
        "label": record.get("label") or "",
        "justification": record.get("justification"),
        "actor": record.get("actor"),
        "timestamp": record.get("timestamp"),
        "evidence": record.get("evidence"),
    }


def decisions_to_records(
    records: Mapping[str, Any] | list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convierte uno o varios registros de decisión para su exportación."""
    if isinstance(records, Mapping):
        records = [records]
    return [decision_to_record(record) for record in records]
