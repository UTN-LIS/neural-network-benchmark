"""Tests for runner module."""

from __future__ import annotations

import csv
import json
import os
import tempfile

import numpy as np
import pandas as pd

from prebenchmark import config, metrics, runner, store
from tests.conftest import WINE_FEATURE_COLUMNS, WINE_CLASSES, WINE_TARGET_COLUMN


def _make_df(n: int = 140, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0, 1, n) for c in WINE_FEATURE_COLUMNS}
    data["free sulfur dioxide"] = rng.integers(0, 80, n).astype(float)
    data["total sulfur dioxide"] = rng.integers(0, 250, n).astype(float)
    data[WINE_TARGET_COLUMN] = np.array([WINE_CLASSES[i % len(WINE_CLASSES)] for i in range(n)])
    return pd.DataFrame(data)


def test_runner_offline_smoke_persists_metadata_and_epochs():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=3, seed=42)
        res = runner.run_experiment(cfg, experiment_id="exp-smoke", store=st, df=df)
        assert res["state"] == "COMPLETED"
        assert res["epochs_executed"] == 3
        run_id = res["run_id"]
        meta_path = os.path.join(td, "exp-smoke", run_id, "metadata.json")
        epochs_path = os.path.join(td, "exp-smoke", run_id, "epochs.csv")
        assert os.path.exists(meta_path)
        assert os.path.exists(epochs_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["execution_state"] == "COMPLETED"
        with open(epochs_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


def test_runner_nan_flagged_and_persisted():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=3, seed=42, lr=1e18)
        res = runner.run_experiment(cfg, experiment_id="exp-nan", store=st, df=df)
        assert res["state"] == "FAILED"
        run_id = res["run_id"]
        meta_path = os.path.join(td, "exp-nan", run_id, "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["execution_state"] == "FAILED"
        assert meta["failure"]["metric"] in {"train_loss", "val_loss", "parameters"}


def test_runner_records_dataset_version_and_environment():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=1, seed=42)
        res = runner.run_experiment(cfg, experiment_id="exp-env", store=st, df=df)
        meta = res["metadata"]
        assert meta["dataset"]["version"].startswith("sha256-")
        assert meta["library_versions"]["python"]
        assert "hardware" in meta
        assert meta["determinism"]["seed"] == 42


def test_runner_early_stopping_triggers_when_criterion_never_improves():
    df = _make_df()
    convergence = config.ConvergenceSpec(
        enabled=True, metric="val_loss", patience=1, min_delta=1e9, mode="min"
    )
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=10, seed=42, convergence=convergence)
        res = runner.run_experiment(cfg, experiment_id="exp-stop", store=st, df=df)
        assert res["state"] == "COMPLETED"
        # First epoch sets the best; the second finds no improvement and stops.
        assert res["epochs_executed"] == 2
        assert res["convergence_trigger"]["reason"] == "early_stopping"
        assert res["convergence_trigger"]["stopped_epoch"] == 2


def test_runner_convergence_disabled_runs_to_max_epochs():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=3, seed=42)
        res = runner.run_experiment(cfg, experiment_id="exp-full", store=st, df=df)
        assert res["epochs_executed"] == 3
        assert res["convergence_trigger"]["reason"] == "disabled"


def test_runner_restore_best_weights_restores_best_epoch():
    df = _make_df()
    # min_delta=1e9 forces the first epoch to be the best; patience=1 stops at epoch 2.
    convergence = config.ConvergenceSpec(
        enabled=True, metric="val_loss", patience=1, min_delta=1e9, mode="min"
    )
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        stopped = runner.run_experiment(
            config.ExperimentConfig(max_epochs=10, seed=42, convergence=convergence),
            experiment_id="exp-restore",
            store=st,
            df=df,
        )
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        single = runner.run_experiment(
            config.ExperimentConfig(max_epochs=1, seed=42),
            experiment_id="exp-single",
            store=st,
            df=df,
        )
    assert stopped["epochs_executed"] == 2
    # restore_best_weights=True => final weights equal the best epoch (epoch 1),
    # so the test-set metrics match a run stopped exactly at epoch 1.
    assert stopped["metadata"]["final_metrics"] == single["metadata"]["final_metrics"]
    assert stopped["metadata"]["convergence"]["restore_best_weights"] is True


def test_runner_records_best_epoch_validation_metrics():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(max_epochs=5, seed=42)
        res = runner.run_experiment(cfg, experiment_id="exp-val", store=st, df=df)
        validation = res["metadata"]["validation_metrics"]
        assert validation["evaluation_set"] == "validation"
        assert 1 <= validation["epoch"] <= 5
        assert validation["loss"] is not None
        assert validation["accuracy"] is not None
        # The recorded val_loss is the minimum across the persisted epoch history.
        epoch_losses = [row["val_loss"] for row in res["epochs"]]
        assert validation["loss"] == min(epoch_losses)


def test_runner_balancing_weighs_training_but_not_evaluation(monkeypatch):
    """La ponderación de clases es un dispositivo de entrenamiento, no de evaluación.

    Si la evaluación reutilizara el criterio ponderado, ``val_loss`` (la señal que
    vigila el early stopping) y ``final_metrics.loss`` serían magnitudes distintas
    en el brazo balanceado y en el no balanceado, y sus diferencias no serían
    comparables.
    """
    df = _make_df()
    train_criteria: list = []
    eval_criteria: list = []
    original_train = metrics.epoch_train_step
    original_eval = metrics.evaluate

    def spy_train(model, data_loader, optimizer, criterion, device="cpu"):
        train_criteria.append(criterion)
        return original_train(model, data_loader, optimizer, criterion, device)

    def spy_eval(model, data_loader, device="cpu", criterion=None):
        eval_criteria.append(criterion)
        return original_eval(model, data_loader, device, criterion)

    monkeypatch.setattr(metrics, "epoch_train_step", spy_train)
    monkeypatch.setattr(metrics, "evaluate", spy_eval)

    cfg = config.ExperimentConfig(
        max_epochs=2,
        seed=42,
        preprocessing=config.PreprocessingSpec(class_weight="balanced"),
    )
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        res = runner.run_experiment(cfg, experiment_id="exp-balanced", store=st, df=df)

    assert res["state"] == "COMPLETED"
    assert train_criteria and eval_criteria
    # El entrenamiento usa la entropía cruzada ponderada por frecuencia de clase.
    assert all(criterion.weight is not None for criterion in train_criteria)
    # La evaluación (validación y prueba) usa la entropía cruzada sin ponderar.
    assert all(criterion.weight is None for criterion in eval_criteria)


def test_runner_balanced_run_records_class_weight_and_unweighted_loss():
    df = _make_df()
    with tempfile.TemporaryDirectory() as td:
        st = store.ResultsStore(td)
        cfg = config.ExperimentConfig(
            max_epochs=2,
            seed=42,
            preprocessing=config.PreprocessingSpec(class_weight="balanced"),
        )
        res = runner.run_experiment(cfg, experiment_id="exp-cw", store=st, df=df)
        meta = res["metadata"]
        assert meta["preprocessing"]["class_weight"] == "balanced"
        assert meta["config_snapshot"]["preprocessing"]["class_weight"] == "balanced"
        assert meta["final_metrics"]["loss"] is not None
        assert meta["final_metrics"]["accuracy"] is not None


