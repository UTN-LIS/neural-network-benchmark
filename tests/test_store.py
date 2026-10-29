"""Unit tests for result persistence, no-overwrite, and the decision log (AC3-6, AC11, AC12)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prebenchmark.config import ExperimentConfig, PreprocessingSpec
from prebenchmark.identity import compute_config_id, compute_space_key
from prebenchmark.space import validate_config
from prebenchmark.store import (
    CONFIG_JSONL_FILE,
    DECISIONS_JSONL_FILE,
    EPOCH_COLUMNS,
    EPOCHS_CSV_FILE,
    RUN_METADATA_FILE,
    ResultsStore,
)

EXPERIMENT_ID = "exp-test-0001"


@pytest.fixture
def store(tmp_path: Path) -> ResultsStore:
    return ResultsStore(tmp_path / "results")


def make_metadata(cfg: ExperimentConfig, run_id: str, **overrides) -> dict:
    metadata = {
        "run_id": run_id,
        "config_id": compute_config_id(cfg),
        "space_key": compute_space_key(cfg),
        "run_type": cfg.run_type,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:01:00Z",
        "config": cfg.to_dict(),
        "dataset": cfg.dataset.to_dict(),
        "architecture": cfg.architecture.to_dict(),
        "activation": cfg.activation,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "init": cfg.init,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "seed": cfg.seed,
        "state": "COMPLETED",
        "warnings": [],
        "failure": None,
        "final_metrics": {"eval_set": "test", "accuracy": 0.5},
    }
    metadata.update(overrides)
    return metadata


def raw_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------


def test_persist_and_load_run_round_trip(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    rows = [
        {"epoch": 1, "train_loss": 1.1, "train_acc": 0.3, "val_loss": 1.0, "val_acc": 0.35},
        {"epoch": 2, "train_loss": 0.9, "train_acc": 0.4, "val_loss": 0.95, "val_acc": 0.38},
    ]
    run_path = store.persist_run(
        EXPERIMENT_ID, "run-1", make_metadata(cfg, "run-1"), rows
    )
    assert (run_path / RUN_METADATA_FILE).exists()
    assert (run_path / EPOCHS_CSV_FILE).exists()

    loaded = store.load_run("run-1")
    assert loaded["experiment_id"] == EXPERIMENT_ID
    assert loaded["metadata"]["state"] == "COMPLETED"
    assert loaded["config_id"] == compute_config_id(cfg)
    assert [row["epoch"] for row in loaded["epochs"]] == [1, 2]
    assert loaded["epochs"][0]["train_loss"] == pytest.approx(1.1)


def test_epochs_csv_uses_canonical_columns(store: ResultsStore) -> None:
    store.persist_run(EXPERIMENT_ID, "run-1", make_metadata(ExperimentConfig(), "run-1"), [])
    header = (
        store.run_dir(EXPERIMENT_ID, "run-1") / EPOCHS_CSV_FILE
    ).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(EPOCH_COLUMNS)


def test_missing_val_metrics_load_as_none(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(
        EXPERIMENT_ID,
        "run-1",
        make_metadata(cfg, "run-1"),
        [{"epoch": 1, "train_loss": 1.0, "train_acc": 0.5}],
    )
    row = store.load_run("run-1")["epochs"][0]
    assert row["val_loss"] is None and row["val_acc"] is None


def test_load_missing_run_raises(store: ResultsStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load_run("does-not-exist")


# ---------------------------------------------------------------------------
# No overwrite / historical preservation / epoch immutability
# ---------------------------------------------------------------------------


def test_re_persisting_same_run_id_fails_explicitly(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(EXPERIMENT_ID, "run-1", make_metadata(cfg, "run-1"))
    with pytest.raises(FileExistsError):
        store.persist_run(EXPERIMENT_ID, "run-1", make_metadata(cfg, "run-1", state="FAILED"))


def test_persisted_epoch_history_is_immutable(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(
        EXPERIMENT_ID,
        "run-1",
        make_metadata(cfg, "run-1"),
        [{"epoch": 1, "train_loss": 1.0, "train_acc": 0.5}],
    )
    epochs_path = store.run_dir(EXPERIMENT_ID, "run-1") / EPOCHS_CSV_FILE
    before = epochs_path.read_bytes()
    with pytest.raises(FileExistsError):
        store.persist_run(
            EXPERIMENT_ID,
            "run-1",
            make_metadata(cfg, "run-1"),
            [{"epoch": 1, "train_loss": 9.9}],
        )
    assert epochs_path.read_bytes() == before


def test_same_config_creates_preserved_distinct_runs(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(EXPERIMENT_ID, "run-a", make_metadata(cfg, "run-a", seed=42))
    store.persist_run(EXPERIMENT_ID, "run-b", make_metadata(cfg, "run-b", seed=7))

    assert store.run_dir(EXPERIMENT_ID, "run-a") != store.run_dir(EXPERIMENT_ID, "run-b")
    assert store.load_run("run-a")["metadata"]["seed"] == 42
    assert store.load_run("run-b")["metadata"]["seed"] == 7
    assert len(store.query_runs(EXPERIMENT_ID)) == 2


# ---------------------------------------------------------------------------
# Failed / partial persistence
# ---------------------------------------------------------------------------


def test_failed_run_persists_with_reason_and_partial_epochs(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    metadata = make_metadata(
        cfg,
        "run-fail",
        state="FAILED",
        failure={"reason": "RuntimeError: exploded", "traceback": "..."},
    )
    store.persist_run(EXPERIMENT_ID, "run-fail", metadata, [{"epoch": 1, "train_loss": float("nan")}])

    loaded = store.load_run("run-fail")
    assert loaded["metadata"]["state"] == "FAILED"
    assert loaded["metadata"]["failure"]["reason"] == "RuntimeError: exploded"
    assert len(loaded["epochs"]) == 1  # partial history preserved, not dropped


def test_failed_run_without_epochs_still_persisted(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(EXPERIMENT_ID, "run-fail", make_metadata(cfg, "run-fail", state="FAILED"))
    loaded = store.load_run("run-fail")
    assert loaded["metadata"]["state"] == "FAILED"
    assert loaded["epochs"] == []


# ---------------------------------------------------------------------------
# Historical queries
# ---------------------------------------------------------------------------


def test_query_runs_filters_by_config_dimensions(store: ResultsStore) -> None:
    base = ExperimentConfig()
    adam = make_metadata(base, "run-adam")
    sgd = make_metadata(
        replace(base, optimizer="sgd", lr=0.01), "run-sgd", optimizer="sgd", lr=0.01
    )
    store.persist_run(EXPERIMENT_ID, "run-adam", adam)
    store.persist_run(EXPERIMENT_ID, "run-sgd", sgd)

    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, optimizer="sgd")] == ["run-sgd"]
    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, lr=0.01)] == ["run-sgd"]
    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, activation="relu")] == [
        "run-adam",
        "run-sgd",
    ]


def test_query_runs_filters_by_state_config_id_and_seed(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    config_id = compute_config_id(cfg)
    store.persist_run(EXPERIMENT_ID, "run-ok", make_metadata(cfg, "run-ok", seed=1))
    store.persist_run(
        EXPERIMENT_ID, "run-bad", make_metadata(cfg, "run-bad", seed=2, state="FAILED")
    )

    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, state="FAILED")] == ["run-bad"]
    assert len(store.query_runs(EXPERIMENT_ID, config_id=config_id)) == 2
    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, seed=2)] == ["run-bad"]


def test_query_runs_accepts_state_and_execution_state_as_aliases(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(
        EXPERIMENT_ID, "run-ok", make_metadata(cfg, "run-ok", execution_state="COMPLETED")
    )
    store.persist_run(
        EXPERIMENT_ID, "run-bad", make_metadata(cfg, "run-bad", execution_state="FAILED")
    )

    # Both spellings resolve to the same canonical field.
    assert [m["run_id"] for m in store.query_runs(EXPERIMENT_ID, state="FAILED")] == ["run-bad"]
    assert [
        m["run_id"] for m in store.query_runs(EXPERIMENT_ID, execution_state="COMPLETED")
    ] == ["run-ok"]


def test_query_runs_filters_by_dataset_and_architecture(store: ResultsStore) -> None:
    base = ExperimentConfig()
    other = replace(
        base,
        dataset=replace(base.dataset, name="other_dataset"),
        architecture=replace(base.architecture, name="other_arch"),
    )
    store.persist_run(EXPERIMENT_ID, "run-wine", make_metadata(base, "run-wine"))
    store.persist_run(EXPERIMENT_ID, "run-other", make_metadata(other, "run-other"))

    assert [
        m["run_id"] for m in store.query_runs(EXPERIMENT_ID, dataset="wine_quality")
    ] == ["run-wine"]
    assert [
        m["run_id"] for m in store.query_runs(EXPERIMENT_ID, architecture="other_arch")
    ] == ["run-other"]


def test_query_runs_across_experiments_when_id_omitted(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run("exp-a", "run-a", make_metadata(cfg, "run-a"))
    store.persist_run("exp-b", "run-b", make_metadata(cfg, "run-b"))
    assert {m["experiment_id"] for m in store.query_runs()} == {"exp-a", "exp-b"}


# ---------------------------------------------------------------------------
# Config registration (append-only, de-duplicated)
# ---------------------------------------------------------------------------


def test_register_config_is_deduplicated_by_config_id(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    first = store.register_config(EXPERIMENT_ID, cfg, validate_config(cfg))
    second = store.register_config(EXPERIMENT_ID, cfg, validate_config(cfg))

    assert first["config_id"] == second["config_id"]
    assert len(raw_lines(store.experiment_dir(EXPERIMENT_ID) / CONFIG_JSONL_FILE)) == 1


def test_register_config_records_validation_state(store: ResultsStore) -> None:
    cfg = replace(ExperimentConfig(), optimizer="adam", lr=0.1)
    record = store.register_config(EXPERIMENT_ID, cfg, validate_config(cfg))
    assert record["validation_state"] == "WARNING"
    assert record["validation_rationale"]  # non-empty


def test_register_config_is_append_only(store: ResultsStore) -> None:
    store.register_config(EXPERIMENT_ID, ExperimentConfig(), validate_config(ExperimentConfig()))
    other = replace(ExperimentConfig(), activation="tanh", init="xavier")
    store.register_config(EXPERIMENT_ID, other, validate_config(other))

    lines = raw_lines(store.experiment_dir(EXPERIMENT_ID) / CONFIG_JSONL_FILE)
    assert len(lines) == 2
    assert json.loads(lines[0])["config"]["activation"] == "relu"  # first entry intact


def test_persist_run_can_register_config(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    store.persist_run(
        EXPERIMENT_ID,
        "run-1",
        make_metadata(cfg, "run-1"),
        config=cfg,
        validation=validate_config(cfg),
    )
    configs = store.load_configs(EXPERIMENT_ID)
    assert len(configs) == 1
    assert configs[0]["config_id"] == compute_config_id(cfg)


def test_configs_jsonl_records_class_weight(store: ResultsStore) -> None:
    cfg = replace(
        ExperimentConfig(),
        preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced"),
    )
    store.register_config(EXPERIMENT_ID, cfg, validate_config(cfg))
    record = store.load_configs(EXPERIMENT_ID)[0]
    assert record["config"]["preprocessing"]["class_weight"] == "balanced"


# ---------------------------------------------------------------------------
# Decision log (append-only, justification required)
# ---------------------------------------------------------------------------


def test_append_decision_requires_justification(store: ResultsStore) -> None:
    with pytest.raises(ValueError):
        store.append_decision(
            EXPERIMENT_ID, decision_type="exclude", justification="   ", actor="analyst", run_id="r"
        )


def test_append_decision_requires_actor(store: ResultsStore) -> None:
    with pytest.raises(ValueError):
        store.append_decision(
            EXPERIMENT_ID, decision_type="keep", justification="good", actor="", run_id="r"
        )


def test_append_decision_requires_entity_reference(store: ResultsStore) -> None:
    with pytest.raises(ValueError):
        store.append_decision(
            EXPERIMENT_ID, decision_type="keep", justification="good", actor="analyst"
        )


def test_decision_history_is_append_only_and_filterable(store: ResultsStore) -> None:
    cfg = ExperimentConfig()
    config_id = compute_config_id(cfg)
    store.append_decision(
        EXPERIMENT_ID,
        decision_type="keep",
        label="VIABLE",
        justification="stable across repetitions",
        actor="analyst",
        run_id="run-1",
        config_id=config_id,
        timestamp="2026-01-01T10:00:00Z",
    )
    store.append_decision(
        EXPERIMENT_ID,
        decision_type="repeat",
        justification="needs more seeds",
        actor="supervisor",
        run_id="run-1",
        timestamp="2026-01-02T10:00:00Z",
    )

    assert len(raw_lines(store.experiment_dir(EXPERIMENT_ID) / DECISIONS_JSONL_FILE)) == 2
    history = store.load_decisions(EXPERIMENT_ID, run_id="run-1")
    assert [d["decision_type"] for d in history] == ["keep", "repeat"]  # order preserved
    assert store.load_decisions(EXPERIMENT_ID, actor="supervisor")[0]["decision_type"] == "repeat"
    assert store.load_decisions(EXPERIMENT_ID, label="VIABLE")[0]["label"] == "VIABLE"
    assert len(store.load_decisions(EXPERIMENT_ID, since="2026-01-02")) == 1
    assert store.load_decisions(EXPERIMENT_ID, group_id="g1") == []


def test_load_decisions_filters_by_type_and_date_range(store: ResultsStore) -> None:
    store.append_decision(
        EXPERIMENT_ID,
        decision_type="keep",
        justification="stable across runs",
        actor="analyst",
        run_id="run-1",
        timestamp="2026-01-01T10:00:00Z",
    )
    store.append_decision(
        EXPERIMENT_ID,
        decision_type="exclude",
        justification="redundant with baseline",
        actor="analyst",
        run_id="run-1",
        timestamp="2026-02-01T10:00:00Z",
    )

    assert [
        d["decision_type"] for d in store.load_decisions(EXPERIMENT_ID, decision_type="keep")
    ] == ["keep"]
    assert [
        d["decision_type"] for d in store.load_decisions(EXPERIMENT_ID, until="2026-01-15")
    ] == ["keep"]


def test_decision_records_are_not_modified(store: ResultsStore) -> None:
    first = store.append_decision(
        EXPERIMENT_ID,
        decision_type="keep",
        justification="original reasoning",
        actor="analyst",
        run_id="run-1",
    )
    store.append_decision(
        EXPERIMENT_ID,
        decision_type="exclude",
        justification="later reasoning",
        actor="analyst",
        run_id="run-1",
    )
    stored = store.load_decisions(EXPERIMENT_ID)[0]
    assert stored["justification"] == "original reasoning"
    assert first["decision_id"] == stored["decision_id"]
