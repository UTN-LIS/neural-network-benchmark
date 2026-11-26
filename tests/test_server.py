"""Tests del esqueleto HTTP: aislamiento, validación, SSE y evidencia."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prebenchmark import config as pb_config
from prebenchmark.config import ConvergenceSpec, ExperimentConfig, PreprocessingSpec
from server import app as server_app
from server.jobs import JobRegistry
from server.serialize import to_jsonable


def _fake_build_config(request: Any) -> ExperimentConfig:
    """Construye la config sin tocar datasets (tests offline y rápidos)."""
    return ExperimentConfig(
        activation=request.activation,
        optimizer=request.optimizer,
        lr=request.lr,
        init=request.init,
        seed=request.seed,
        max_epochs=request.max_epochs,
        preprocessing=PreprocessingSpec(
            class_weight=None if request.class_weight == "none" else request.class_weight
        ),
        convergence=ConvergenceSpec(enabled=request.early_stop),
    )


def _fake_run_experiment(
    cfg: ExperimentConfig,
    experiment_id: str = "exp-test",
    store: Any = None,
    *,
    df: Any = None,
    on_epoch: Any = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    assert store is not None, "la UI debe persistir la corrida en su store aislado"
    for epoch in range(1, 4):
        record = {
            "run_id": "run-ui-test",
            "config_id": "cfg-ui-test",
            "epoch": epoch,
            "train_loss": 2.0 / epoch,
            "train_acc": 0.4 + epoch * 0.1,
            "val_loss": 2.2 / epoch,
            "val_acc": 0.35 + epoch * 0.1,
            "epoch_time": 0.01,
        }
        if on_epoch is not None:
            on_epoch(record)

    metadata = {
        "run_id": "run-ui-test",
        "config_id": "cfg-ui-test",
        "execution_state": "COMPLETED",
        "warnings": [],
        "config_snapshot": cfg.to_dict(),
        "validation_metrics": {
            "evaluation_set": "validation",
            "epoch": 3,
            "loss": 0.73,
            "accuracy": 0.65,
            "balanced_accuracy": 0.62,
            "precision_macro": 0.64,
            "recall_macro": 0.63,
            "f1_macro": 0.63,
            "confusion_matrix": [[1, 0], [0, 1]],
        },
        "final_metrics": {
            "evaluation_set": "test",
            "loss": 0.70,
            "accuracy": 0.66,
            "balanced_accuracy": 0.63,
            "precision_macro": 0.65,
            "recall_macro": 0.64,
            "f1_macro": 0.64,
            "confusion_matrix": [[1, 0], [0, 1]],
        },
        "training_time": 0.03,
        "n_params": 1814,
    }
    return {
        "run_id": "run-ui-test",
        "config_id": "cfg-ui-test",
        "state": "COMPLETED",
        "epochs_executed": 3,
        "convergence_trigger": {"reason": "early_stopping", "best_epoch": 3},
        "metadata": metadata,
        "epochs": [],
    }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_app, "_build_config", _fake_build_config)
    monkeypatch.setattr(server_app.runner, "run_experiment", _fake_run_experiment)
    application = server_app.create_app(
        store_root=tmp_path / "ui-store",
        experiment_id="ui-test",
        exports_root=tmp_path / "exports",
        reports_root=tmp_path / "reports",
    )
    with TestClient(application) as test_client:
        yield test_client


def _payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset": "wine_quality",
        "activation": "relu",
        "optimizer": "adam",
        "lr": 0.01,
        "init": "he",
        "class_weight": "none",
        "seed": 42,
        "max_epochs": 50,
        "early_stop": True,
        "include_test": False,
        "device": "auto",
    }
    base.update(overrides)
    return base


def _sse_events(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = None
    with client.stream("GET", f"/api/runs/{job_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
                if event_name:
                    payload["_event"] = event_name
                events.append(payload)
                event_name = None
    return events


def test_health_declares_single_worker_and_isolated_store(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["max_workers"] == 1
    assert body["experiment_id"] == "ui-test"
    assert body["store_root"].endswith("ui-store")


def test_options_exposes_the_four_dimensions_and_protocol(client: TestClient) -> None:
    body = client.get("/api/options").json()
    assert "wine_quality" in [dataset["name"] for dataset in body["datasets"]]
    assert body["activations"] == ["relu", "sigmoid", "tanh"]
    assert body["optimizers"] == ["sgd", "adam"]
    assert body["learning_rates"] == [0.001, 0.01, 0.1]
    assert body["initializations"] == ["xavier", "he"]
    assert body["protocol"]["early_stopping"]["enabled_by_default"] is True
    assert body["protocol"]["batch_size"] == 32


def test_options_exposes_only_the_curated_dataset_menu(client: TestClient) -> None:
    body = client.get("/api/options").json()
    names = [dataset["name"] for dataset in body["datasets"]]
    assert names == list(pb_config.DATASET_MENU)
    assert "digits" not in names


def test_validate_returns_domain_state(client: TestClient) -> None:
    invalid = client.post("/api/validate", json=_payload(activation="softmax")).json()
    assert invalid["state"] == "INVALID"
    assert invalid["errors"]

    warning = client.post(
        "/api/validate", json=_payload(activation="tanh", init="he", optimizer="sgd", lr=0.001)
    ).json()
    assert warning["state"] == "WARNING"
    assert any("He/Kaiming" in message for message in warning["warnings"])


def test_run_rejects_invalid_configuration(client: TestClient) -> None:
    response = client.post("/api/runs", json=_payload(activation="softmax"))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["validation"]["state"] == "INVALID"


def test_run_streams_epochs_and_hides_test_by_default(client: TestClient) -> None:
    created = client.post("/api/runs", json=_payload())
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    events = _sse_events(client, job_id)
    epochs = [event for event in events if event.get("_event") == "epoch"]
    done = [event for event in events if event.get("_event") == "done"]

    assert [epoch["epoch"] for epoch in epochs] == [1, 2, 3]
    assert len(done) == 1
    result = done[0]["result"]
    assert result["execution_state"] == "COMPLETED"
    assert result["validation_metrics"]["f1_macro"] == pytest.approx(0.63)
    assert result["test_metrics"] is None
    assert result["test_hidden"] is True
    assert "final_metrics" not in result["metadata"]
    assert result["space_key"].startswith("space-v1-")
    assert result["group_key"].startswith("grp-v1-")
    assert result["n_params"] == 1814


def test_run_includes_test_only_when_explicitly_requested(client: TestClient) -> None:
    job_id = client.post("/api/runs", json=_payload(include_test=True)).json()["job_id"]
    done = [event for event in _sse_events(client, job_id) if event.get("_event") == "done"][0]
    result = done["result"]
    assert result["test_metrics"]["evaluation_set"] == "test"
    assert result["test_metrics"]["f1_macro"] == pytest.approx(0.64)
    assert result["test_hidden"] is False


def test_job_status_is_queryable(client: TestClient) -> None:
    job_id = client.post("/api/runs", json=_payload()).json()["job_id"]
    _sse_events(client, job_id)
    status = client.get(f"/api/runs/{job_id}").json()
    assert status["status"] == "completed"
    assert status["epochs"] == 3
    assert status["execution_state"] == "COMPLETED"
    assert client.get("/api/runs/job-inexistente").status_code == 404


def test_cancel_marks_the_run_incomplete(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_run(
        cfg: ExperimentConfig,
        experiment_id: str = "exp-test",
        store: Any = None,
        *,
        df: Any = None,
        on_epoch: Any = None,
        should_stop: Any = None,
    ) -> dict[str, Any]:
        emitted = 0
        for epoch in range(1, 500):
            if should_stop is not None and should_stop():
                metadata = {
                    "run_id": "run-cancel",
                    "config_id": "cfg-cancel",
                    "execution_state": "INCOMPLETE",
                    "warnings": ["run cancelled by user request"],
                    "config_snapshot": cfg.to_dict(),
                    "validation_metrics": {"evaluation_set": "validation", "epoch": emitted},
                    "final_metrics": {"evaluation_set": "test"},
                    "training_time": 0.1,
                    "n_params": 10,
                }
                return {
                    "run_id": "run-cancel",
                    "config_id": "cfg-cancel",
                    "state": "INCOMPLETE",
                    "epochs_executed": emitted,
                    "convergence_trigger": {"reason": "cancelled_by_user"},
                    "metadata": metadata,
                    "epochs": [],
                }
            emitted = epoch
            if on_epoch is not None:
                on_epoch(
                    {
                        "run_id": "run-cancel",
                        "config_id": "cfg-cancel",
                        "epoch": epoch,
                        "train_loss": 1.0 / epoch,
                        "train_acc": 0.5,
                        "val_loss": 1.0 / epoch,
                        "val_acc": 0.5,
                        "epoch_time": 0.01,
                    }
                )
            time.sleep(0.02)
        raise AssertionError("la corrida no fue cancelada")

    monkeypatch.setattr(server_app.runner, "run_experiment", slow_run)
    job_id = client.post("/api/runs", json=_payload(max_epochs=499)).json()["job_id"]

    for _ in range(200):
        if client.get(f"/api/runs/{job_id}").json()["epochs"] >= 2:
            break
        time.sleep(0.01)

    cancelled = client.post(f"/api/runs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True

    events = _sse_events(client, job_id)
    done = [event for event in events if event.get("_event") == "done"][0]
    assert done["result"]["execution_state"] == "INCOMPLETE"
    assert done["result"]["epochs_executed"] < 499
    assert done["result"]["convergence_trigger"]["reason"] == "cancelled_by_user"


def test_cancel_is_idempotent_and_404_for_unknown_job(client: TestClient) -> None:
    job_id = client.post("/api/runs", json=_payload()).json()["job_id"]
    _sse_events(client, job_id)
    response = client.post(f"/api/runs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert client.post("/api/runs/job-inexistente/cancel").status_code == 404


def test_evidence_derives_failure_modes_from_exports(client: TestClient, tmp_path: Path) -> None:
    exports = tmp_path / "exports" / "exp-demo"
    exports.mkdir(parents=True)
    rows = []
    families = [
        ("adam", 0.1, 30, 0.19, 4.0),
        ("sgd", 0.001, 500, 0.21, 50.0),
        ("adam", 0.001, 110, 0.88, 12.0),
        ("adam", 0.01, 40, 0.88, 5.0),
        ("sgd", 0.01, 300, 0.86, 30.0),
        ("sgd", 0.1, 90, 0.87, 10.0),
    ]
    config_ids = []
    for family_index, (optimizer, lr, epochs, f1, seconds) in enumerate(families):
        for seed in (42, 123):
            config_id = f"cfg-{family_index}-{seed}"
            config_ids.append(config_id)
            rows.append(
                {
                    "config_id": config_id,
                    "dataset": "wine_quality",
                    "optimizer": optimizer,
                    "lr": lr,
                    "f1_macro": f1,
                    "accuracy": f1 + 0.05,
                    "epochs_executed": epochs,
                    "max_epochs": 500,
                    "training_time": seconds,
                    "execution_state": "COMPLETED",
                    "seed": seed,
                }
            )
    pd.DataFrame(rows).to_csv(exports / "master_20260101T000000Z.csv", index=False)
    pd.DataFrame({"config_id": config_ids, "class_weight": [None] * len(config_ids)}).to_csv(
        exports / "configs_20260101T000000Z.csv", index=False
    )
    pd.DataFrame({"group_key": [f"grp-{index}" for index in range(6)]}).to_csv(
        exports / "aggregates_20260101T000000Z.csv", index=False
    )

    # Un experimento de prueba sobre un dataset interno no debe aparecer en la
    # comparación general: el menú curado filtra por DATASET_MENU.
    internal = tmp_path / "exports" / "exp-internal"
    internal.mkdir()
    pd.DataFrame(
        [
            {
                "config_id": "cfg-digits",
                "dataset": "digits",
                "optimizer": "adam",
                "lr": 0.001,
                "epochs_executed": 50,
                "f1_macro": 0.97,
                "accuracy": 0.97,
                "training_time": 1.0,
                "seed": 42,
                "execution_state": "COMPLETED",
            }
        ]
    ).to_csv(internal / "master_20260101T000000Z.csv", index=False)

    body = client.get("/api/evidence?dataset=wine_quality").json()
    assert body["available"] is True
    assert body["experiment_id"] == "exp-demo"
    assert body["runs"] == 12
    assert body["groups"] == 6
    assert len(body["families"]) == 6

    modes = {issue["family"]: issue["failure_mode"] for issue in body["known_issues"]}
    assert modes["adam + lr=0.1"] == "collapse"
    assert modes["sgd + lr=0.001"] == "no_converge"

    catalog = client.get("/api/evidence").json()
    assert catalog["datasets_with_suite"] == ["wine_quality"]
    assert all(item["dataset"] != "digits" for item in catalog["experiments"])


def test_reports_are_served_without_path_traversal(client: TestClient, tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "exp-demo"
    report_dir.mkdir(parents=True)
    (report_dir / "analysis_20260101T000000Z.html").write_text(
        "<html>informe</html>", encoding="utf-8"
    )

    response = client.get("/api/reports/exp-demo")
    assert response.status_code == 200
    assert "informe" in response.text

    assert client.get("/api/reports/..%2F..%2Fetc").status_code == 404


def _write_history_run(store_root: Path, run_id: str, start_time: str, **overrides: Any) -> Path:
    """Crea un run persistido mínimo con el mismo layout que ``persist_run``."""
    run_dir = store_root / "ui-test" / run_id
    run_dir.mkdir(parents=True)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "config_id": "cfg-history",
        "experiment_id": "ui-test",
        "start_time": start_time,
        "end_time": start_time,
        "dataset": {"name": "wine_quality", "target": "quality"},
        "architecture": {"name": "wine_quality_mlp", "hidden_layers": [32, 16]},
        "activation": "relu",
        "optimizer": "adam",
        "lr": 0.01,
        "init": "he",
        "seed": 42,
        "batch_size": 32,
        "max_epochs": 50,
        "epochs_executed": 4,
        "preprocessing": {"scaler": "standard", "class_weight": None},
        "execution_state": "COMPLETED",
        "warnings": [],
        "validation_metrics": {
            "evaluation_set": "validation",
            "epoch": 3,
            "loss": 0.7,
            "accuracy": 0.66,
            "balanced_accuracy": 0.63,
            "precision_macro": 0.65,
            "recall_macro": 0.64,
            "f1_macro": 0.64,
        },
        "final_metrics": {"evaluation_set": "test", "f1_macro": 0.99},
        "training_time": 1.5,
        "n_params": 1814,
    }
    metadata.update(overrides)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_history_lists_persisted_runs_newest_first(client: TestClient, tmp_path: Path) -> None:
    store_root = tmp_path / "ui-store"
    _write_history_run(store_root, "20260101T000000Z-aaaaaa", "20260101T000000Z")
    _write_history_run(
        store_root,
        "20260102T000000Z-bbbbbb",
        "20260102T000000Z",
        execution_state="INCOMPLETE",
        warnings=["run cancelled by user request"],
    )
    (store_root / "ui-test" / "sin-metadata").mkdir()  # se ignora: no es una corrida

    body = client.get("/api/history").json()
    assert body["experiment_id"] == "ui-test"
    assert body["total"] == 2
    assert body["returned"] == 2
    assert [row["run_id"] for row in body["runs"]] == [
        "20260102T000000Z-bbbbbb",
        "20260101T000000Z-aaaaaa",
    ]

    newest = body["runs"][0]
    assert newest["dataset"] == "wine_quality"
    assert newest["execution_state"] == "INCOMPLETE"
    assert newest["f1_macro"] == pytest.approx(0.64)
    assert newest["class_weight"] is None
    assert newest["warning_count"] == 1
    assert "final_metrics" not in json.dumps(body)

    limited = client.get("/api/history", params={"limit": 1}).json()
    assert limited["total"] == 2
    assert limited["returned"] == 1
    assert [row["run_id"] for row in limited["runs"]] == ["20260102T000000Z-bbbbbb"]


def test_history_detail_returns_epochs_and_hides_test_metrics(
    client: TestClient, tmp_path: Path
) -> None:
    store_root = tmp_path / "ui-store"
    run_dir = _write_history_run(store_root, "20260103T000000Z-cccccc", "20260103T000000Z")
    (run_dir / "epochs.csv").write_text(
        "run_id,config_id,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time\n"
        "20260103T000000Z-cccccc,cfg-history,1,1.0,0.5,0.9,0.5,0.1\n",
        encoding="utf-8",
    )

    body = client.get("/api/history/20260103T000000Z-cccccc").json()
    assert body["run_id"] == "20260103T000000Z-cccccc"
    assert body["metadata"]["validation_metrics"]["f1_macro"] == pytest.approx(0.64)
    assert "final_metrics" not in body["metadata"]
    assert body["test_hidden"] is True
    assert body["epochs"][0]["val_loss"] == pytest.approx(0.9)

    assert client.get("/api/history/inexistente").status_code == 404


def test_history_delete_removes_only_the_selected_run(client: TestClient, tmp_path: Path) -> None:
    store_root = tmp_path / "ui-store"
    doomed = _write_history_run(store_root, "20260106T000000Z-dddddd", "20260106T000000Z")
    survivor = _write_history_run(store_root, "20260107T000000Z-eeeeee", "20260107T000000Z")
    stray = store_root / "ui-test" / "sin-metadata"
    stray.mkdir()

    response = client.delete("/api/history/20260106T000000Z-dddddd")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert not doomed.exists()
    assert survivor.exists()
    assert stray.exists()  # un directorio que no es una corrida no se elimina

    remaining = client.get("/api/history").json()
    assert [row["run_id"] for row in remaining["runs"]] == ["20260107T000000Z-eeeeee"]
    assert client.delete("/api/history/20260106T000000Z-dddddd").status_code == 404
    assert client.delete("/api/history/..%2F..%2Fetc").status_code == 404


def test_registry_refuses_more_than_one_worker() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        JobRegistry(lambda job: None, max_workers=2)


def test_to_jsonable_makes_nan_and_numpy_safe() -> None:
    payload = to_jsonable(
        {"nan": float("nan"), "numpy": np.float64(1.5), "array": np.array([1, 2])}
    )
    assert payload == {"nan": None, "numpy": 1.5, "array": [1, 2]}


def test_dataset_detail_is_404_for_unknown_dataset(client: TestClient) -> None:
    assert client.get("/api/datasets/no_existe").status_code == 404


def test_options_shape_matches_config_registry(client: TestClient) -> None:
    body = client.get("/api/options").json()
    assert body["activations"] == list(pb_config.KNOWN_ACTIVATIONS)
    assert body["optimizers"] == list(pb_config.KNOWN_OPTIMIZERS)
    assert body["initializations"] == list(pb_config.KNOWN_INITIALIZATIONS)
