"""API HTTP y servidor de la UI de prebenchmark.

Decisiones que el esqueleto hace explícitas
-------------------------------------------
* **Store aislado.** Las corridas de la UI escriben en ``results/ui-runs`` por
  defecto, nunca en el store de las suites (``results/experiments``). Un job de
  la UI no puede contaminar ``exp-sweep-*``.
* **Un solo worker.** Ver :mod:`server.jobs`: el seeding del runner es global al
  proceso. ``max_workers=1`` preserva la reproducibilidad.
* **Test set reservado.** La respuesta principal expone
  ``validation_metrics``; ``final_metrics`` (test) solo se devuelve si el cliente
  envía ``include_test=true``, y queda identificado como tal.
* **Early stopping encendido por defecto en la UI**, porque las suites se
  corrieron así y sin él una corrida puede agotar las 500 épocas (~122 s medidos).
* **La UI no reimplementa el dominio.** Construye ``ExperimentConfig``, valida con
  ``space.validate_config`` y ejecuta con ``runner.run_experiment``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prebenchmark import config as pb_config
from prebenchmark import data as pb_data
from prebenchmark import runner, space, store
from prebenchmark.aggregate import compute_group_key
from prebenchmark.config import ConvergenceSpec, ExperimentConfig
from prebenchmark.space import compute_space_key

from . import evidence
from .jobs import Job, JobRegistry
from .schemas import JobCreated, JobStatus, RunRequest, ValidationOut
from .serialize import to_jsonable

MAX_WORKERS = 1
STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_STORE_ROOT = Path(os.environ.get("PREBENCHMARK_UI_STORE", "results/ui-runs"))
DEFAULT_EXPERIMENT_ID = os.environ.get("PREBENCHMARK_UI_EXPERIMENT", "ui-exploratory")
DEFAULT_EXPORTS_ROOT = Path(os.environ.get("PREBENCHMARK_EXPORTS", "results/exports"))
DEFAULT_REPORTS_ROOT = Path(os.environ.get("PREBENCHMARK_REPORTS", "results/reports"))


def _configure_torch_threads() -> None:
    """Fija los hilos intra-operación de torch, si el operador lo pide.

    Con un único worker, el valor por defecto de torch (todos los núcleos) es el
    correcto. La variable de entorno existe para evitar oversubscription cuando
    este proceso convive con otras cargas.
    """
    requested = int(os.environ.get("PREBENCHMARK_TORCH_THREADS", "0") or 0)
    if requested <= 0:
        return
    try:
        import torch

        torch.set_num_threads(requested)
    except Exception:  # noqa: BLE001 - configuración opcional, nunca bloquea el arranque
        pass


def _dataset_payload(name: str, *, force_load: bool) -> dict[str, Any]:
    """Metadatos de un dataset registrado.

    ``force_load=False`` (listado) solo lee datasets ya cacheados: seleccionar un
    dataset no debe disparar una descarga sorpresa. ``force_load=True`` (detalle)
    sí resuelve el dataset, que es una acción explícita del usuario.
    """
    definition = pb_data.get_dataset(name)
    cache_path = Path.cwd() / pb_data.DEFAULT_CACHE_SUBDIR / definition.cache_filename
    cached = cache_path.exists()
    payload: dict[str, Any] = {
        "name": definition.name,
        "source": definition.source,
        "target": definition.target,
        "cached": cached,
        "n_samples": None,
        "n_features": None,
        "n_classes": None,
        "imbalance": None,
        "hidden_layers": [32, 16],
    }
    if not cached and not force_load:
        return payload

    info = pb_data.describe_dataset(name)
    frame = pd.read_csv(info.cache_path, sep=";")
    counts = frame[definition.target].value_counts()
    payload.update(
        {
            "n_samples": int(info.n_samples),
            "n_features": int(info.n_features),
            "n_classes": int(info.n_classes),
            "imbalance": round(float(counts.max() / counts.min()), 2) if len(counts) else None,
            "cached": True,
        }
    )
    return payload


def _build_config(request: RunRequest) -> ExperimentConfig:
    cfg = pb_data.dataset_base_config(request.dataset)
    return replace(
        cfg,
        activation=request.activation,
        optimizer=request.optimizer,
        lr=request.lr,
        init=request.init,
        seed=request.seed,
        max_epochs=request.max_epochs,
        device=request.device,
        preprocessing=replace(
            cfg.preprocessing,
            class_weight=None if request.class_weight == "none" else request.class_weight,
        ),
        convergence=ConvergenceSpec(enabled=request.early_stop),
    )


def _result_payload(result: dict[str, Any], job: Job) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    validation_metrics = dict(metadata.get("validation_metrics") or {})
    final_metrics = metadata.get("final_metrics")
    if not job.include_test:
        # El set de prueba no se devuelve salvo pedido explícito. La copia
        # persistida en el store conserva el dato completo para trazabilidad.
        metadata.pop("final_metrics", None)

    payload = {
        "run_id": result.get("run_id"),
        "config_id": result.get("config_id"),
        "execution_state": result.get("state"),
        "validation_metrics": validation_metrics,
        "test_metrics": final_metrics if job.include_test else None,
        "test_hidden": not job.include_test,
        "epochs_executed": result.get("epochs_executed"),
        "convergence_trigger": result.get("convergence_trigger"),
        "training_time": metadata.get("training_time"),
        "n_params": metadata.get("n_params"),
        "warnings": metadata.get("warnings", []),
        "config": metadata.get("config_snapshot") or job.config.to_dict(),
        "space_key": compute_space_key(job.config),
        "group_key": compute_group_key(metadata),
        "metadata": metadata,
    }
    return to_jsonable(payload)


def _history_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """Proyección compacta de ``metadata.json`` para la tabla de historial.

    Solo usa datos ya presentes en los metadatos: no abre ``epochs.csv``, así
    que listar cientos de corridas sigue siendo barato. ``final_metrics`` (set
    de prueba) se omite a propósito, igual que en la corrida en vivo.
    """
    dataset = metadata.get("dataset") or {}
    architecture = metadata.get("architecture") or {}
    preprocessing = metadata.get("preprocessing") or {}
    validation = metadata.get("validation_metrics") or {}
    warnings = metadata.get("warnings") or []
    return {
        "run_id": metadata.get("run_id"),
        "config_id": metadata.get("config_id"),
        "experiment_id": metadata.get("experiment_id"),
        "start_time": metadata.get("start_time"),
        "end_time": metadata.get("end_time"),
        "execution_state": metadata.get("execution_state") or metadata.get("state"),
        "dataset": dataset.get("name") if isinstance(dataset, dict) else dataset,
        "activation": metadata.get("activation"),
        "optimizer": metadata.get("optimizer"),
        "lr": metadata.get("lr"),
        "init": metadata.get("init"),
        "seed": metadata.get("seed"),
        "class_weight": (
            preprocessing.get("class_weight") if isinstance(preprocessing, dict) else None
        ),
        "hidden_layers": (
            architecture.get("hidden_layers") if isinstance(architecture, dict) else None
        ),
        "max_epochs": metadata.get("max_epochs"),
        "epochs_executed": metadata.get("epochs_executed"),
        "f1_macro": validation.get("f1_macro"),
        "accuracy": validation.get("accuracy"),
        "balanced_accuracy": validation.get("balanced_accuracy"),
        "val_loss": validation.get("loss"),
        "best_epoch": validation.get("epoch"),
        "training_time": metadata.get("training_time"),
        "n_params": metadata.get("n_params"),
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
    }


class Service:
    """Estado de larga vida del proceso: store aislado y cola de un worker."""

    def __init__(
        self,
        *,
        store_root: Path,
        experiment_id: str,
        exports_root: Path,
        reports_root: Path,
    ) -> None:
        self.store = store.ResultsStore(store_root)
        self.store_root = store_root
        self.experiment_id = experiment_id
        self.exports_root = exports_root
        self.reports_root = reports_root
        self.registry = JobRegistry(self._execute, max_workers=MAX_WORKERS)

    def _execute(self, job: Job) -> None:
        job.mark_running()
        try:
            result = runner.run_experiment(
                job.config,
                experiment_id=self.experiment_id,
                store=self.store,
                on_epoch=job.publish_epoch,
                should_stop=job.should_stop,
            )
            job.mark_done(_result_payload(result, job))
        except Exception as error:  # noqa: BLE001 - se reporta al cliente por SSE
            job.mark_failed(f"{type(error).__name__}: {error}")


def _service(request: Request) -> Service:
    return request.app.state.service


def _validation_out(validation: space.ValidationResult) -> ValidationOut:
    return ValidationOut(
        state=validation.state,
        rationale=list(validation.rationale),
        warnings=list(validation.warnings),
        errors=list(validation.errors),
    )


def create_app(
    *,
    store_root: Path | None = None,
    experiment_id: str | None = None,
    exports_root: Path | None = None,
    reports_root: Path | None = None,
) -> FastAPI:
    resolved_store = store_root or DEFAULT_STORE_ROOT
    resolved_experiment = experiment_id or DEFAULT_EXPERIMENT_ID
    resolved_exports = exports_root or DEFAULT_EXPORTS_ROOT
    resolved_reports = reports_root or DEFAULT_REPORTS_ROOT

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        _configure_torch_threads()
        application.state.service = Service(
            store_root=resolved_store,
            experiment_id=resolved_experiment,
            exports_root=resolved_exports,
            reports_root=resolved_reports,
        )
        try:
            yield
        finally:
            application.state.service.registry.shutdown()

    application = FastAPI(
        title="Prebenchmark UI",
        version="0.1.0",
        description="Corrida individual de configuraciones y comparación de suites completas.",
        lifespan=lifespan,
    )

    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        service = _service(request)
        return {
            "status": "ok",
            "max_workers": MAX_WORKERS,
            "experiment_id": service.experiment_id,
            "store_root": str(service.store_root),
        }

    @application.get("/api/options")
    def options() -> dict[str, Any]:
        parameter_space = space.ParameterSpace()
        return {
            "datasets": [
                _dataset_payload(name, force_load=False) for name in pb_config.DATASET_MENU
            ],
            "activations": list(pb_config.KNOWN_ACTIVATIONS),
            "optimizers": list(pb_config.KNOWN_OPTIMIZERS),
            "learning_rates": list(parameter_space.learning_rates),
            "initializations": list(pb_config.KNOWN_INITIALIZATIONS),
            "class_weights": ["none", "balanced"],
            "protocol": {
                "batch_size": pb_config.DEFAULT_BATCH_SIZE,
                "max_epochs": pb_config.DEFAULT_MAX_EPOCHS,
                "split": {
                    "train": pb_config.DEFAULT_TRAIN_RATIO,
                    "val": pb_config.DEFAULT_VAL_RATIO,
                    "test": pb_config.DEFAULT_TEST_RATIO,
                },
                "early_stopping": {
                    "enabled_by_default": True,
                    "metric": pb_config.DEFAULT_CONVERGENCE_METRIC,
                    "patience": pb_config.DEFAULT_CONVERGENCE_PATIENCE,
                    "min_delta": pb_config.DEFAULT_CONVERGENCE_MIN_DELTA,
                },
            },
        }

    @application.get("/api/datasets/{name}")
    def dataset_detail(name: str) -> dict[str, Any]:
        if name not in pb_data.DATASETS:
            raise HTTPException(status_code=404, detail=f"dataset desconocido: {name}")
        try:
            return _dataset_payload(name, force_load=True)
        except Exception as error:  # noqa: BLE001 - fallo de red/caché al cargar
            raise HTTPException(
                status_code=502, detail=f"No se pudo cargar el dataset: {error}"
            ) from error

    @application.post("/api/validate")
    def validate(request: RunRequest) -> dict[str, Any]:
        try:
            cfg = _build_config(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - fallo de red/caché al cargar
            raise HTTPException(
                status_code=502, detail=f"No se pudo cargar el dataset: {error}"
            ) from error
        return space.validate_config(cfg).to_dict()

    @application.post("/api/runs", status_code=202, response_model=JobCreated)
    def create_run(request: RunRequest, http_request: Request) -> JobCreated:
        service = _service(http_request)
        try:
            cfg = _build_config(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - fallo de red/caché al cargar
            raise HTTPException(
                status_code=502, detail=f"No se pudo cargar el dataset: {error}"
            ) from error

        validation = space.validate_config(cfg)
        if not validation.is_executable():
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "La configuración es inválida y no se puede ejecutar",
                    "validation": validation.to_dict(),
                },
            )

        job = Job(
            dataset=request.dataset,
            config=cfg,
            include_test=request.include_test,
            validation=validation,
        )
        service.registry.submit(job)
        return JobCreated(job_id=job.id, status=job.status, validation=_validation_out(validation))

    @application.get("/api/runs", response_model=list[JobStatus])
    def list_runs(request: Request, limit: int = 20) -> list[JobStatus]:
        service = _service(request)
        return [JobStatus(**snapshot) for snapshot in service.registry.recent(limit=limit)]

    @application.get("/api/runs/{job_id}", response_model=JobStatus)
    def run_status(job_id: str, request: Request) -> JobStatus:
        job = _service(request).registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la corrida")
        return JobStatus(**job.snapshot())

    @application.post("/api/runs/{job_id}/cancel", response_model=JobStatus)
    def cancel_run(job_id: str, request: Request) -> JobStatus:
        job = _service(request).registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la corrida")
        if not job.done:
            job.request_cancel()
        return JobStatus(**job.snapshot())

    @application.get("/api/runs/{job_id}/events")
    async def run_events(job_id: str, request: Request) -> StreamingResponse:
        job = _service(request).registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la corrida")

        async def generate() -> AsyncIterator[str]:
            index = 0
            last_ping = time.monotonic()
            while True:
                events, done = job.events_from(index)
                for event in events:
                    index += 1
                    yield (
                        f"id: {index}\n"
                        f"event: {event['type']}\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
                if done:
                    break
                if time.monotonic() - last_ping >= 15:
                    last_ping = time.monotonic()
                    yield ": ping\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/api/history")
    def run_history(request: Request, limit: int = 100) -> dict[str, Any]:
        """Corridas persistidas en disco, de la más reciente a la más antigua.

        A diferencia de ``GET /api/runs`` (estado en memoria del proceso), este
        listado se reconstruye desde ``results/ui-runs/<experiment_id>/`` y por
        lo tanto sobrevive a reinicios del servidor.
        """
        service = _service(request)
        records = service.store.query_runs(service.experiment_id)
        records.reverse()  # query_runs ordena ascendente por start_time
        bounded = max(1, min(limit, 500))
        return to_jsonable(
            {
                "experiment_id": service.experiment_id,
                "store_root": str(service.store_root),
                "total": len(records),
                "returned": min(len(records), bounded),
                "runs": [_history_row(metadata) for metadata in records[:bounded]],
            }
        )

    @application.get("/api/history/{run_id}")
    def run_history_detail(run_id: str, request: Request) -> dict[str, Any]:
        """Detalle de una corrida persistida: metadatos completos + épocas."""
        service = _service(request)
        try:
            record = service.store.load_run(run_id, service.experiment_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="No se encontró la corrida") from error

        metadata = dict(record.get("metadata") or {})
        # Misma política que la corrida en vivo: el set de prueba queda
        # persistido para trazabilidad, pero no se expone por defecto.
        metadata.pop("final_metrics", None)
        return to_jsonable(
            {
                "run_id": record.get("run_id"),
                "experiment_id": record.get("experiment_id"),
                "config_id": record.get("config_id"),
                "metadata": metadata,
                "epochs": record.get("epochs") or [],
                "path": record.get("path"),
                "test_hidden": True,
            }
        )

    @application.delete("/api/history/{run_id}")
    def delete_history_run(run_id: str, request: Request) -> dict[str, Any]:
        """Elimina una corrida persistida. La acción es irreversible."""
        service = _service(request)
        try:
            result = service.store.delete_run(service.experiment_id, run_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="No se encontró la corrida") from error
        except OSError as error:
            raise HTTPException(
                status_code=500, detail=f"No se pudo eliminar la corrida: {error}"
            ) from error
        return {"deleted": True, **result}

    @application.get("/api/evidence")
    def get_evidence(request: Request, dataset: str | None = None) -> dict[str, Any]:
        service = _service(request)
        experiments = evidence.load_experiments(
            service.exports_root,
            service.reports_root,
            datasets=pb_config.DATASET_MENU,
        )
        if dataset is not None:
            return evidence.dataset_evidence(dataset, experiments)
        return {
            "experiments": experiments,
            "datasets_with_suite": sorted({item["dataset"] for item in experiments}),
        }

    @application.get("/api/reports/{experiment_id}", include_in_schema=False)
    def report_html(experiment_id: str, request: Request) -> FileResponse:
        path = evidence.report_file(_service(request).reports_root, experiment_id, "html")
        if path is None:
            raise HTTPException(status_code=404, detail="No se encontró el informe")
        return FileResponse(path, media_type="text/html")

    @application.get("/api/reports/{experiment_id}/pdf", include_in_schema=False)
    def report_pdf(experiment_id: str, request: Request) -> FileResponse:
        path = evidence.report_file(_service(request).reports_root, experiment_id, "pdf")
        if path is None:
            raise HTTPException(status_code=404, detail="No se encontró el informe")
        return FileResponse(path, media_type="application/pdf")

    return application


app = create_app()
