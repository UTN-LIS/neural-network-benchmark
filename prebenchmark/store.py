"""Persistencia de resultados por experimento con una garantía estructural de no sobrescritura.

Disposición en disco (bajo una raíz configurable, por defecto ``results/``)::

    <root>/<experiment-id>/
        experiment.json          # metadatos del experimento (línea base, justificación, ...)
        configs.jsonl            # solo adición, deduplicado por config_id
        decisions.jsonl          # registro de decisiones manual de solo adición
        <run_id>/
            metadata.json        # registro completo e inmutable de la ejecución
            epochs.csv           # historial inmutable por época

Cada directorio de ejecución se crea con :func:`os.mkdir`, que es atómico y lanza
``FileExistsError`` si el directorio ya existe. Por lo tanto, "nunca sobrescribir una
ejecución persistida" es estructural: una ejecución se escribe una vez y no existe
API para modificarla. Los dos registros JSONL son de solo adición -- las líneas existentes
nunca se reescriben. Nada aquí depende de la salida de consola ni del estado en memoria.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .identity import compute_config_id, compute_space_key, new_decision_id, utc_timestamp
from .space import ValidationResult

__all__ = [
    "CONFIG_JSONL_FILE",
    "DECISIONS_JSONL_FILE",
    "EPOCHS_CSV_FILE",
    "EPOCH_COLUMNS",
    "EXPERIMENT_METADATA_FILE",
    "RUN_METADATA_FILE",
    "ResultsStore",
]

EXPERIMENT_METADATA_FILE = "experiment.json"
CONFIG_JSONL_FILE = "configs.jsonl"
DECISIONS_JSONL_FILE = "decisions.jsonl"
RUN_METADATA_FILE = "metadata.json"
EPOCHS_CSV_FILE = "epochs.csv"
EXPORTS_DIR_NAME = "exports"

# Orden de columnas de ``epochs.csv`` (columnas de validación vacías/NaN cuando están ausentes).
EPOCH_COLUMNS: tuple[str, ...] = (
    "run_id",
    "config_id",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "epoch_time",
)

# Campos de metadatos expuestos por los filtros de ``query_runs``.
_CONFIG_FILTER_FIELDS = (
    "run_type",
    "activation",
    "optimizer",
    "lr",
    "init",
    "loss",
    "batch_size",
    "max_epochs",
    "seed",
    "device",
)


class ResultsStore:
    """Persistencia basada en archivos para experimentos, ejecuciones y decisiones.

    Parámetros
    ----------
    root:
        Directorio bajo el cual se crean las carpetas de experimentos (por defecto
        ``results``, que está en gitignore y se crea en tiempo de ejecución).
    """

    def __init__(self, root: str | os.PathLike[str] = "results") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Rutas
    # ------------------------------------------------------------------

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def run_dir(self, experiment_id: str, run_id: str) -> Path:
        return self.experiment_dir(experiment_id) / run_id

    def _configs_path(self, experiment_id: str) -> Path:
        return self.experiment_dir(experiment_id) / CONFIG_JSONL_FILE

    def _decisions_path(self, experiment_id: str) -> Path:
        return self.experiment_dir(experiment_id) / DECISIONS_JSONL_FILE

    # ------------------------------------------------------------------
    # Helpers de E/S de bajo nivel
    # ------------------------------------------------------------------

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        """Escribe ``text`` en ``path`` una sola vez, mediante un archivo temporal + reemplazo atómico."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _atomic_mkdir(path: Path) -> None:
        """Crea ``path`` de forma atómica; lanza ``FileExistsError`` si ya existe."""
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path)

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Registro de configuración (solo adición, deduplicado)
    # ------------------------------------------------------------------

    def register_config(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        validation: ValidationResult | None = None,
        *,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Añade una definición de configuración a ``configs.jsonl`` una vez por ``config_id``.

        Volver a registrar la misma configuración (mismo ``config_id``) no tiene efecto
        y devuelve el registro existente -- el registro es de solo adición y nunca
        se reescribe.
        """
        config_id = compute_config_id(config)
        for existing in self._read_jsonl(self._configs_path(experiment_id)):
            if existing.get("config_id") == config_id:
                return existing

        record: dict[str, Any] = {
            "config_id": config_id,
            "space_key": compute_space_key(config),
            "config": config.to_dict(),
            "preprocessing": config.preprocessing.to_dict(),
            "convergence": config.convergence.to_dict(),
            "validation_state": validation.state if validation is not None else None,
            "validation_rationale": list(validation.rationale) if validation is not None else [],
            "timestamp": timestamp or utc_timestamp(),
        }
        self._append_jsonl(self._configs_path(experiment_id), record)
        return record

    def load_configs(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._read_jsonl(self._configs_path(experiment_id))

    # ------------------------------------------------------------------
    # Persistencia de ejecuciones (escritura única por run_id)
    # ------------------------------------------------------------------

    def persist_run(
        self,
        experiment_id: str,
        run_id: str,
        metadata: Mapping[str, Any],
        epochs: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: ExperimentConfig | None = None,
        validation: ValidationResult | None = None,
    ) -> Path:
        """Persiste una ejecución sin sobrescribir nunca un ``run_id`` existente.

        Crea ``<experiment>/<run_id>/`` de forma atómica, escribe ``metadata.json``
        y ``epochs.csv`` exactamente una vez, y devuelve el directorio de la ejecución. Lanza
        ``FileExistsError`` si la ejecución ya existe. Opcionalmente registra primero la
        instantánea de configuración (solo adición, deduplicada).
        """
        if config is not None:
            self.register_config(experiment_id, config, validation)

        run_path = self.run_dir(experiment_id, run_id)
        self._atomic_mkdir(run_path)

        payload = dict(metadata)
        payload.setdefault("run_id", run_id)
        payload.setdefault("experiment_id", experiment_id)
        self._write_text_atomic(
            run_path / RUN_METADATA_FILE,
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        )
        self._write_epochs(run_path / EPOCHS_CSV_FILE, run_id, payload.get("config_id"), epochs)
        return run_path

    def _write_epochs(
        self,
        path: Path,
        run_id: str,
        config_id: str | None,
        epochs: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EPOCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in epochs or ():
            record = dict(row)
            record.setdefault("run_id", run_id)
            record.setdefault("config_id", config_id)
            writer.writerow({column: record.get(column) for column in EPOCH_COLUMNS})
        self._write_text_atomic(path, buffer.getvalue())

    def load_run(self, run_id: str, experiment_id: str | None = None) -> dict[str, Any]:
        """Devuelve el registro completo de ``run_id`` (metadatos + historial de épocas)."""
        for exp_id, run_path in self._iter_run_paths(experiment_id):
            if run_path.name == run_id:
                metadata = self._read_json(run_path / RUN_METADATA_FILE)
                return {
                    "experiment_id": exp_id,
                    "run_id": run_path.name,
                    "config_id": metadata.get("config_id"),
                    "metadata": metadata,
                    "epochs": self._read_epochs(run_path / EPOCHS_CSV_FILE),
                    "path": str(run_path),
                }
        raise FileNotFoundError(f"run '{run_id}' not found")

    @staticmethod
    def _read_epochs(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [ResultsStore._coerce_epoch_row(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _coerce_epoch_row(row: Mapping[str, Any]) -> dict[str, Any]:
        coerced: dict[str, Any] = {}
        for key, value in row.items():
            if value in (None, "", "NA"):
                coerced[key] = None
                continue
            number: float | int
            try:
                number = int(value)
            except (TypeError, ValueError):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    coerced[key] = value
                    continue
            coerced[key] = number
        return coerced

    def _iter_run_paths(self, experiment_id: str | None = None) -> Iterator[tuple[str, Path]]:
        if experiment_id is not None:
            exp_dirs = [self.experiment_dir(experiment_id)]
        elif self.root.exists():
            exp_dirs = [p for p in self.root.iterdir() if p.is_dir() and p.name != EXPORTS_DIR_NAME]
        else:
            exp_dirs = []
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            for child in sorted(exp_dir.iterdir()):
                if child.is_dir() and (child / RUN_METADATA_FILE).exists():
                    yield exp_dir.name, child

    # ------------------------------------------------------------------
    # Consultas históricas (nunca dependen del estado en memoria del proceso)
    # ------------------------------------------------------------------

    def query_runs(self, experiment_id: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """Devuelve los metadatos de ejecución que coinciden con todos los filtros proporcionados.

        Filtros admitidos: ``run_id``, ``config_id``, ``state``, ``run_type``,
        ``dataset``, ``architecture``, y las dimensiones de configuración
        ``activation``/``optimizer``/``lr``/``init``/``batch_size``/``seed``/
        ``loss``/``max_epochs``/``device``.
        """
        results: list[dict[str, Any]] = []
        for exp_id, run_path in self._iter_run_paths(experiment_id):
            metadata = self._read_json(run_path / RUN_METADATA_FILE)
            metadata.setdefault("experiment_id", exp_id)
            if self._matches(metadata, filters):
                results.append(metadata)
        results.sort(key=lambda m: (m.get("start_time") or "", m.get("run_id") or ""))
        return results

    @staticmethod
    def _matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            if expected is None:
                continue
            if ResultsStore._resolve_field(metadata, key) != expected:
                return False
        return True

    @staticmethod
    def _resolve_field(metadata: Mapping[str, Any], key: str) -> Any:
        config = metadata.get("config", {}) or {}
        if key in ("state", "execution_state"):
            # ``execution_state`` es el campo canónico; ``state`` se acepta como
            # alias en ambos lados para que los registros heredados y actuales coincidan.
            value = metadata.get("execution_state")
            if value is None:
                value = metadata.get("state")
            return value if value is not None else config.get(key)
        if key == "dataset":
            value = metadata.get("dataset")
            if isinstance(value, Mapping):
                value = value.get("name")
            return value if value is not None else (config.get("dataset", {}) or {}).get("name")
        if key == "architecture":
            value = metadata.get("architecture")
            if isinstance(value, Mapping):
                value = value.get("name")
            return (
                value if value is not None else (config.get("architecture", {}) or {}).get("name")
            )
        if key in _CONFIG_FILTER_FIELDS:
            value = metadata.get(key)
            return value if value is not None else config.get(key)
        return metadata.get(key)

    # ------------------------------------------------------------------
    # Registro de decisiones (solo adición; se requiere justificación)
    # ------------------------------------------------------------------

    def append_decision(
        self,
        experiment_id: str,
        *,
        decision_type: str,
        justification: str,
        actor: str,
        label: str | None = None,
        run_id: str | None = None,
        config_id: str | None = None,
        group_id: str | None = None,
        evidence: Any = None,
        timestamp: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        """Añade un registro de decisión manual. Las entradas existentes nunca se modifican.

        ``justification`` y ``actor`` deben ser no vacíos, y se requiere al menos una
        referencia de entidad (``run_id``/``config_id``/``group_id``).
        Esto es únicamente andamiaje manual -- no ocurre ninguna clasificación automática.
        """
        if not str(justification).strip():
            raise ValueError("decision justification must be a non-empty string")
        if not str(actor).strip():
            raise ValueError("decision actor must be a non-empty string")
        entity = {
            name: value
            for name, value in (
                ("run_id", run_id),
                ("config_id", config_id),
                ("group_id", group_id),
            )
            if value
        }
        if not entity:
            raise ValueError("a decision must reference a run_id, config_id or group_id")

        record: dict[str, Any] = {
            "decision_id": decision_id or new_decision_id(),
            "entity": entity,
            "decision_type": decision_type,
            "label": label,
            "justification": justification,
            "actor": actor,
            "timestamp": timestamp or utc_timestamp(),
            "evidence": evidence,
        }
        self._append_jsonl(self._decisions_path(experiment_id), record)
        return record

    def load_decisions(
        self, experiment_id: str | None = None, **filters: Any
    ) -> list[dict[str, Any]]:
        """Devuelve los registros de decisiones en orden de adición, opcionalmente filtrados.

        Filtros admitidos: ``run_id``/``config_id``/``group_id`` (comparados en la
        entidad), ``decision_type``, ``label``, ``actor``, y los límites de marca de tiempo
        inclusivos ``since``/``until``.
        """
        if experiment_id is not None:
            exp_ids = [experiment_id]
        elif self.root.exists():
            exp_ids = [
                p.name for p in self.root.iterdir() if p.is_dir() and p.name != EXPORTS_DIR_NAME
            ]
        else:
            exp_ids = []

        records: list[dict[str, Any]] = []
        for exp_id in exp_ids:
            records.extend(self._read_jsonl(self._decisions_path(exp_id)))
        return [record for record in records if self._matches_decision(record, filters)]

    @staticmethod
    def _matches_decision(record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        entity = record.get("entity", {}) or {}
        for key, expected in filters.items():
            if expected is None:
                continue
            if key in ("run_id", "config_id", "group_id"):
                if entity.get(key) != expected:
                    return False
            elif key == "since":
                if (record.get("timestamp") or "") < str(expected):
                    return False
            elif key == "until":
                if (record.get("timestamp") or "") > str(expected):
                    return False
            elif record.get(key) != expected:
                return False
        return True

    # ------------------------------------------------------------------
    # Metadatos opcionales del experimento (selección de línea base / justificación)
    # ------------------------------------------------------------------

    def write_experiment_metadata(self, experiment_id: str, metadata: Mapping[str, Any]) -> Path:
        path = self.experiment_dir(experiment_id) / EXPERIMENT_METADATA_FILE
        self._write_text_atomic(path, json.dumps(dict(metadata), indent=2) + "\n")
        return path

    def load_experiment_metadata(self, experiment_id: str) -> dict[str, Any]:
        path = self.experiment_dir(experiment_id) / EXPERIMENT_METADATA_FILE
        if not path.exists():
            return {}
        return self._read_json(path)
