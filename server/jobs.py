"""Ejecución de corridas individuales en un único hilo de trabajo.

Por qué ``max_workers=1`` (decisión, no limitación pendiente)
------------------------------------------------------------
``prebenchmark.runner.run_experiment`` siembra los RNG **globales del proceso**
(``np.random.seed`` y ``torch.manual_seed``) al comenzar cada corrida, porque de
ahí dependen la partición train/val/test y la inicialización de pesos. Dos
entrenamientos concurrentes en el mismo proceso se pisarían ese estado global y
el resultado dejaría de ser reproducible.

Un solo worker garantiza que cada corrida vea exactamente el RNG que sembró.
El paralelismo entre corridas independientes (barridos) debe resolverse con
procesos, no con hilos: cada proceso tiene su propio estado global.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from prebenchmark.config import ExperimentConfig
from prebenchmark.space import ValidationResult

from .serialize import to_jsonable

MAX_RETAINED_JOBS = 50
MAX_RETAINED_EVENTS = 2000


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    """Estado observable de una corrida individual.

    ``events`` es un log append-only leído por SSE. Se guarda completo (hasta un
    tope) para que un cliente que se reconecta pueda reproducir desde el inicio
    en lugar de perder las épocas ya emitidas.
    """

    dataset: str
    config: ExperimentConfig
    include_test: bool
    validation: ValidationResult
    id: str = field(default_factory=lambda: "job-" + uuid.uuid4().hex[:12])
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    execution_state: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    epochs_seen: int = 0
    cancel_requested: bool = False
    _events: list[dict[str, Any]] = field(default_factory=list)
    _done: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # -- escritura -----------------------------------------------------
    def _append(self, event: dict[str, Any]) -> None:
        with self._lock:
            if len(self._events) < MAX_RETAINED_EVENTS:
                self._events.append(event)
            if event.get("type") == "epoch":
                self.epochs_seen += 1

    def publish_epoch(self, record: dict[str, Any]) -> None:
        self._append({"type": "epoch", **to_jsonable(record)})

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = utc_now()
        self._append({"type": "status", "status": "running", "dataset": self.dataset})

    def mark_done(self, result: dict[str, Any]) -> None:
        self.status = "completed"
        self.finished_at = utc_now()
        self.execution_state = result.get("execution_state")
        self.result = result
        with self._lock:
            self._events.append({"type": "done", "result": result})
            self._done = True

    def mark_failed(self, message: str) -> None:
        self.status = "failed"
        self.finished_at = utc_now()
        self.error = message
        with self._lock:
            self._events.append({"type": "failed", "message": message})
            self._done = True

    # -- lectura -------------------------------------------------------
    def request_cancel(self) -> None:
        """Pide la cancelación cooperativa; el runner la consulta por época."""
        self.cancel_requested = True
        self._cancel.set()

    def should_stop(self) -> bool:
        return self._cancel.is_set()

    def events_from(self, index: int) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            return list(self._events[index:]), self._done

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "epochs": self.epochs_seen,
            "execution_state": self.execution_state,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "result": self.result,
        }


class JobRegistry:
    """Pool de un solo worker + registro en memoria de los últimos jobs."""

    def __init__(self, execute: Callable[[Job], None], max_workers: int = 1) -> None:
        if max_workers != 1:
            raise ValueError(
                "max_workers must be 1: concurrent runs would race on the process-global RNG"
            )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="prebenchmark-ui"
        )
        self._execute = execute
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > MAX_RETAINED_JOBS:
                self._jobs.pop(self._order.popleft(), None)
        self._executor.submit(self._execute, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._order)[-limit:]
            jobs = [self._jobs[job_id] for job_id in ids if job_id in self._jobs]
        return [job.snapshot() for job in reversed(jobs)]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
