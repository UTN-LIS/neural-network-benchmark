"""Ejecutor de entrenamiento reproducible para experimentos."""

from __future__ import annotations

import contextlib
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import data, environment, metrics, model
from .config import (
    DEFAULT_CONVERGENCE_MIN_DELTA,
    DEFAULT_CONVERGENCE_PATIENCE,
    DEFAULT_MAX_EPOCHS,
    ExperimentConfig,
    resolve_convergence,
)
from .identity import compute_config_id, new_run_id, utc_timestamp
from .space import validate_config
from .store import ResultsStore

# Campo canónico para el estado de ejecución duradero de una ejecución.
EXECUTION_STATE_FIELD = "execution_state"


def _select_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _improved(value: float, best: float, min_delta: float, mode: str) -> bool:
    """Verdadero cuando ``value`` supera a ``best`` por más de ``min_delta``."""
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


def _convergence_trigger(
    *,
    enabled: bool,
    metric_name: str | None,
    patience: int | None,
    min_delta: float | None,
    mode: str | None,
    reason: str,
    best_epoch: int | None = None,
    best_value: float | None = None,
    stopped_epoch: int | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric_name,
        "patience": patience,
        "min_delta": min_delta,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "stopped_epoch": stopped_epoch,
        "reason": reason,
    }


def _determinism_note(seed: int) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "seed": seed,
        "numpy_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": cuda,
        "note": (
            "seeds fix software RNGs; cuDNN kernels may still be non-deterministic"
            if cuda
            else "CPU-only run; seeds fully control the software RNGs"
        ),
    }


def run_experiment(
    cfg: ExperimentConfig,
    experiment_id: str = "exp-test",
    store: ResultsStore | None = None,
    *,
    df: Any = None,
) -> dict[str, Any]:
    """Ejecuta el experimento de forma reproducible. Devuelve un dict similar a RunResult."""
    # Siembra determinista
    seed = cfg.seed if cfg.seed is not None else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _select_device(cfg.device)
    run_id = new_run_id()
    config_id = compute_config_id(cfg)
    start_time = time.time()
    start_ts = utc_timestamp()

    validation = validate_config(cfg)

    if store is not None:
        store.register_config(experiment_id, cfg, validation)

    # Resuelve el criterio de convergencia concreto (los valores predeterminados
    # documentados completan cualquier campo no especificado) para registrar
    # textualmente lo que realmente se aplicó.
    resolved_convergence = (
        resolve_convergence(cfg.convergence) if cfg.convergence.enabled else cfg.convergence
    )
    convergence_meta = resolved_convergence.to_dict()
    convergence_meta["restore_best_weights"] = True
    metric_name = resolved_convergence.metric if cfg.convergence.enabled else None
    patience = (
        resolved_convergence.patience
        if resolved_convergence.patience is not None
        else DEFAULT_CONVERGENCE_PATIENCE
    )
    min_delta = (
        resolved_convergence.min_delta
        if resolved_convergence.min_delta is not None
        else DEFAULT_CONVERGENCE_MIN_DELTA
    )
    mode = resolved_convergence.mode

    # Huella del dataset (de mejor esfuerzo: un fallo nunca aborta la ejecución).
    warnings: list[str] = list(validation.rationale) if validation.rationale else []
    dataset_meta = cfg.dataset.to_dict()
    try:
        dataset_meta["version"] = data.resolve_dataset_version(df, cfg.dataset.name)
    except Exception as error:  # pragma: no cover - defensivo: problemas de red/caché
        warnings.append(f"dataset version unavailable: {error}")

    # Captura del entorno (de mejor esfuerzo).
    try:
        env = environment.capture_environment()
    except Exception as error:  # pragma: no cover - defensivo
        env = {"hardware": {}, "library_versions": {}}
        warnings.append(f"environment capture skipped: {error}")

    epoch_records: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch: int | None = None
    best_state_dict: dict | None = None
    restore_best_weights = True
    epochs_without_improvement = 0
    convergence_metric_unavailable = False
    stopped_reason = "disabled" if not cfg.convergence.enabled else "max_epochs"
    current_epoch = 0
    best_val_loss: float | None = None
    best_val_loss_epoch: int | None = None
    best_val_acc_at_best: float | None = None
    failure_context: dict[str, Any] = {"epoch": None, "batch": None, "metric": None}

    try:
        dataloaders = data.build_dataloaders(cfg, df=df)
        train_loader = dataloaders["train_loader"]
        val_loader = dataloaders["val_loader"]
        test_loader = dataloaders["test_loader"]

        m = model.build_model(cfg=cfg)
        m = model.initialize_weights(m, cfg.init, cfg.activation)
        m.to(device)

        if cfg.optimizer.lower() == "adam":
            optimizer: optim.Optimizer = optim.Adam(m.parameters(), lr=cfg.lr)
        else:
            optimizer = optim.SGD(m.parameters(), lr=cfg.lr)

        # La ponderación de clases es un dispositivo de ENTRENAMIENTO: el objetivo se
        # pondera para forzar al modelo a aprender las clases minoritarias, pero la
        # evaluación (validación y prueba) usa SIEMPRE entropía cruzada sin ponderar.
        # De lo contrario ``val_loss`` --la señal que vigila el early stopping-- y
        # ``final_metrics.loss`` serían magnitudes distintas en cada brazo y sus
        # diferencias no serían comparables entre un brazo balanceado y otro que no lo está.
        criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.CrossEntropyLoss()

        if cfg.preprocessing.class_weight == "balanced":
            splits = data.build_splits(cfg, df=df)
            y_train = splits["y_train"]
            classes, counts = np.unique(y_train, return_counts=True)
            # Vector indexado por identificador de clase (no por posición): una clase
            # ausente en train no puede desalinear la longitud del vector de pesos.
            weights = torch.ones(cfg.architecture.output_dim, dtype=torch.float32)
            class_ids = torch.from_numpy(classes.astype(np.int64))
            weights[class_ids] = torch.tensor(counts.max() / counts, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        max_epochs = cfg.max_epochs if cfg.max_epochs is not None else DEFAULT_MAX_EPOCHS
        for epoch in range(1, max_epochs + 1):
            current_epoch = epoch
            epoch_start = time.time()
            train_res = metrics.epoch_train_step(m, train_loader, optimizer, criterion, device)
            if train_res["non_finite"]:
                failure_context = {
                    "epoch": epoch,
                    "batch": train_res["non_finite_batch"],
                    "metric": train_res["non_finite_source"],
                }
                raise ValueError(
                    f"non-finite validation detected in {train_res['non_finite_source']} "
                    f"at epoch {epoch}, batch {train_res['non_finite_batch']}"
                )
            val_res: dict[str, Any] = {}
            if val_loader is not None:
                val_eval = metrics.evaluate(m, val_loader, device, eval_criterion)
                if val_eval["non_finite"]:
                    failure_context = {
                        "epoch": epoch,
                        "batch": None,
                        "metric": "val_loss",
                    }
                    raise ValueError(f"non-finite validation loss detected at epoch {epoch}")
                val_res = {"val_loss": val_eval["loss"], "val_acc": val_eval["accuracy"]}
                # Mejor época por val_loss: métrica de comparación de la etapa exploratoria.
                if np.isfinite(val_res["val_loss"]) and (
                    best_val_loss is None or val_res["val_loss"] < best_val_loss
                ):
                    best_val_loss = val_res["val_loss"]
                    best_val_loss_epoch = epoch
                    best_val_acc_at_best = val_res["val_acc"]
            if not model.parameters_are_finite(m):
                failure_context = {"epoch": epoch, "batch": None, "metric": "parameters"}
                raise ValueError(f"non-finite model parameters detected at epoch {epoch}")

            record = {
                "run_id": run_id,
                "config_id": config_id,
                "epoch": epoch,
                "train_loss": train_res["train_loss"],
                "train_acc": train_res["train_acc"],
                **val_res,
                "epoch_time": time.time() - epoch_start,
            }
            epoch_records.append(record)

            # Convergencia / parada temprana (solo cuando se habilita explícitamente).
            if cfg.convergence.enabled and not convergence_metric_unavailable:
                value = (
                    train_res["train_loss"]
                    if metric_name == "train_loss"
                    else val_res.get("val_loss")
                )
                if value is None or not np.isfinite(value):
                    convergence_metric_unavailable = True
                    stopped_reason = "metric_unavailable"
                    warnings.append(
                        f"convergence metric '{metric_name}' unavailable; running to max_epochs"
                    )
                elif best_value is None or _improved(value, best_value, min_delta, mode):
                    best_value = value
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        stopped_reason = "early_stopping"
                        break

        # Restaura los mejores pesos (restore_best_weights=True, semǭntica de Keras):
        # se conserva la Ǹpoca con mejor mǸtrica de convergencia, no la ǧltima.
        if restore_best_weights and best_state_dict is not None:
            m.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            best_state_dict = None

        # Evaluación final sobre prueba (siempre sin ponderar, igual que la validación)
        test_eval = metrics.evaluate(m, test_loader, device, eval_criterion)
        if test_eval["non_finite"]:
            failure_context = {"epoch": current_epoch, "batch": None, "metric": "test_loss"}
            raise ValueError("non-finite test loss detected during final evaluation")
        test_eval.pop("non_finite", None)
        test_eval.pop("non_finite_source", None)

        end_time = time.time()
        end_ts = utc_timestamp()
        state = "COMPLETED"
        if validation.state == "WARNING":
            state = "COMPLETED_WITH_WARNINGS"

        metadata = {
            "run_id": run_id,
            "config_id": config_id,
            "experiment_id": experiment_id,
            "run_type": cfg.run_type,
            "start_time": start_ts,
            "end_time": end_ts,
            "config_snapshot": cfg.to_dict(),
            "dataset": dataset_meta,
            "architecture": cfg.architecture.to_dict(),
            "seed": seed,
            "activation": cfg.activation,
            "optimizer": cfg.optimizer,
            "lr": cfg.lr,
            "init": cfg.init,
            "batch_size": cfg.batch_size,
            "max_epochs": max_epochs,
            "epochs_executed": len(epoch_records),
            "loss": cfg.loss,
            "split": cfg.split.to_dict(),
            "convergence": convergence_meta,
            "convergence_trigger": _convergence_trigger(
                enabled=cfg.convergence.enabled,
                metric_name=metric_name,
                patience=patience,
                min_delta=min_delta,
                mode=mode,
                reason=stopped_reason,
                best_epoch=best_epoch,
                best_value=best_value,
                stopped_epoch=epoch_records[-1]["epoch"] if epoch_records else None,
            ),
            "preprocessing": cfg.preprocessing.to_dict(),
            "device": device,
            "hardware": env["hardware"],
            "library_versions": env["library_versions"],
            "determinism": _determinism_note(seed),
            EXECUTION_STATE_FIELD: state,
            "warnings": list(warnings),
            "final_metrics": {"evaluation_set": "test", **test_eval},
            "validation_metrics": {
                "evaluation_set": "validation",
                "epoch": best_val_loss_epoch,
                "loss": best_val_loss,
                "accuracy": best_val_acc_at_best,
            },
            "n_params": sum(p.numel() for p in m.parameters()),
            "training_time": end_time - start_time,
        }

        if store is not None:
            store.persist_run(
                experiment_id,
                run_id,
                metadata,
                epochs=epoch_records,
                config=cfg,
                validation=validation,
            )

        return {
            "run_id": run_id,
            "config_id": config_id,
            "state": state,
            "epochs_executed": len(epoch_records),
            "convergence_trigger": metadata["convergence_trigger"],
            "metadata": metadata,
            "epochs": epoch_records,
        }
    except Exception as e:
        end_time = time.time()
        end_ts = utc_timestamp()
        state = "FAILED"
        failure: dict[str, Any] = {
            "reason": str(e),
            "traceback": traceback.format_exc(),
            "epoch": failure_context["epoch"],
            "batch": failure_context["batch"],
            "metric": failure_context["metric"],
        }
        metadata = {
            "run_id": run_id,
            "config_id": config_id,
            "experiment_id": experiment_id,
            "run_type": cfg.run_type,
            "start_time": start_ts,
            "end_time": end_ts,
            "config_snapshot": cfg.to_dict(),
            "dataset": dataset_meta,
            EXECUTION_STATE_FIELD: state,
            "failure": failure,
            "epochs_executed": len(epoch_records),
            "device": device,
        }
        if store is not None:
            # Un fallo de persistencia no debe sobrescribir el resultado del entrenamiento.
            with contextlib.suppress(Exception):
                store.persist_run(
                    experiment_id,
                    run_id,
                    metadata,
                    epochs=epoch_records,
                    config=cfg,
                    validation=validation,
                )
        return {
            "run_id": run_id,
            "config_id": config_id,
            "state": state,
            "failure": failure,
            "epochs_executed": len(epoch_records),
            "metadata": metadata,
        }
