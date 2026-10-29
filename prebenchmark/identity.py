"""Identificadores estables y únicos para configuraciones y ejecuciones de prebenchmark.

Dos identificadores están deliberadamente separados:

* ``config_id`` -- un hash de contenido sobre la configuración *completa* (cada campo,
  incluido ``class_weight``). Cualquier campo diferente produce un id diferente, por lo que
  puede usarse para agrupar repeticiones del mismo experimento.
* ``space_key`` -- un hash de contenido sobre solo las cuatro dimensiones libres del espacio
  (``activation``/``optimizer``/``lr``/``init``). ``class_weight`` y los ajustes del protocolo
  nunca dividen el espacio de 36 combinaciones.

``compute_space_key`` ya reside en :mod:`prebenchmark.space`; se importa
y se reexporta aquí para una única superficie de identidad -- NO se redefine. Los
identificadores de ejecución se generan con una marca de tiempo UTC más un sufijo aleatorio y son el
nombre de directorio de una ejecución persistida, lo que hace que "nunca sobrescribir" sea estructural.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import re
from datetime import datetime, timezone

from .config import ExperimentConfig
from .space import compute_space_key, space_key_dims  # reexportado (no redefinido)

__all__ = [
    "CONFIG_ID_PREFIX",
    "EXPERIMENT_ID_PREFIX",
    "RUN_ID_PATTERN",
    "canonical_config_payload",
    "compute_config_id",
    "compute_space_key",
    "new_decision_id",
    "new_experiment_id",
    "new_run_id",
    "space_key_dims",
    "utc_timestamp",
]

CONFIG_ID_PREFIX = "cfg-v1-"
EXPERIMENT_ID_PREFIX = "exp-"
DECISION_ID_PREFIX = "dec-"

_HASH_LENGTH = 12
# 3 bytes aleatorios -> 6 caracteres hexadecimales en minúscula.
_RANDOM_BYTES = 3
_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def utc_timestamp(moment: datetime | None = None) -> str:
    """Devuelve una marca de tiempo UTC ordenable lexicográficamente (``...Z``)."""
    current = moment or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime(_STAMP_FORMAT)


def canonical_config_payload(cfg: ExperimentConfig) -> str:
    """JSON canónico de la configuración *completa* (orden de claves estable)."""
    return json.dumps(
        cfg.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_config_id(cfg: ExperimentConfig) -> str:
    """Hash de contenido estable sobre cada campo de la configuración.

    Determinista entre procesos y máquinas; configuraciones idénticas producen el
    mismo id, y una diferencia en cualquier campo (incluido ``class_weight``) produce
    uno diferente.
    """
    digest = hashlib.sha256(
        canonical_config_payload(cfg).encode("utf-8")
    ).hexdigest()[:_HASH_LENGTH]
    return f"{CONFIG_ID_PREFIX}{digest}"


def _slugify(label: str) -> str:
    slug = _SLUG_PATTERN.sub("-", label.strip().lower()).strip("-")
    return slug or "experiment"


def new_run_id(timestamp: str | None = None) -> str:
    """Devuelve un id de ejecución único: ``<UTC timestamp>-<6 hex chars>``.

    El sufijo aleatorio mantiene los ids únicos incluso para ejecuciones iniciadas en el mismo segundo.
    """
    stamp = timestamp or utc_timestamp()
    return f"{stamp}-{secrets.token_hex(_RANDOM_BYTES)}"


def new_experiment_id(label: str | None = None, timestamp: str | None = None) -> str:
    """Devuelve un id de experimento único, opcionalmente etiquetado con un slug."""
    stamp = timestamp or utc_timestamp()
    suffix = secrets.token_hex(_RANDOM_BYTES)
    if label:
        return f"{EXPERIMENT_ID_PREFIX}{_slugify(label)}-{stamp}-{suffix}"
    return f"{EXPERIMENT_ID_PREFIX}{stamp}-{suffix}"


def new_decision_id(timestamp: str | None = None) -> str:
    """Devuelve un id de decisión manual único (clave de entrada del registro de solo adición)."""
    stamp = timestamp or utc_timestamp()
    return f"{DECISION_ID_PREFIX}{stamp}-{secrets.token_hex(_RANDOM_BYTES)}"
