"""Tests for the CSV export layer (``prebenchmark.export``).

These are pure unit/ integration tests over a real :class:`ResultsStore`, with
hand-built run records shaped like the runner output (``execution_state`` for
COMPLETED, ``state`` for FAILED, ``config_snapshot`` for the embedded config).
No training happens here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
from prebenchmark import aggregate, export, identity, space, store
from prebenchmark.config import ExperimentConfig

EXPERIMENT = "exp-export"

DEFAULT_METRICS = {
    "accuracy": 0.80,
    "balanced_accuracy": 0.78,
    "precision_macro": 0.77,
    "recall_macro": 0.76,
    "f1_macro": 0.75,
    "loss": 0.50,
}


def _metadata(
    cfg: ExperimentConfig,
    *,
    run_id: str,
    state: str = "COMPLETED",
    metrics: dict | None = None,
    warnings: list[str] | None = None,
    failure: str | None = None,
    start_time: str = "2026-01-01T00:00:00Z",
    training_time: float = 1.5,
) -> dict:
    config_id = identity.compute_config_id(cfg)
    meta = {
        "run_id": run_id,
        "config_id": config_id,
        "experiment_id": EXPERIMENT,
        "run_type": cfg.run_type,
        "start_time": start_time,
        "end_time": start_time,
        "config_snapshot": cfg.to_dict(),
        "dataset": cfg.dataset.to_dict(),
        "architecture": cfg.architecture.to_dict(),
        "seed": cfg.seed,
        "activation": cfg.activation,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "init": cfg.init,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "epochs_executed": 2,
        "loss": cfg.loss,
        "split": cfg.split.to_dict(),
        "convergence": cfg.convergence.to_dict(),
        "preprocessing": cfg.preprocessing.to_dict(),
        "device": "cpu",
        "n_params": 987,
        "training_time": training_time,
        "warnings": list(warnings or []),
    }
    if state == "FAILED":
        meta["state"] = "FAILED"
        meta["failure"] = {"reason": failure or "boom"}
        meta["epochs_executed"] = 0
    else:
        meta["execution_state"] = state
        meta["final_metrics"] = {"evaluation_set": "test", **(metrics or DEFAULT_METRICS)}
    return meta


def _epochs(run_id: str, config_id: str, count: int = 2) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "config_id": config_id,
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "train_acc": 0.5 + epoch * 0.05,
            "val_loss": 0.9 / epoch,
            "val_acc": 0.4 + epoch * 0.05,
            "epoch_time": 0.01,
        }
        for epoch in range(1, count + 1)
    ]


def _build_store(root: Path) -> store.ResultsStore:
    results = store.ResultsStore(root)
    base = ExperimentConfig(max_epochs=3, seed=42)

    def persist(cfg: ExperimentConfig, run_id: str, **kwargs) -> None:
        meta = _metadata(cfg, run_id=run_id, **kwargs)
        state = meta.get("execution_state") or meta.get("state")
        epochs = [] if state == "FAILED" else _epochs(run_id, meta["config_id"])
        results.persist_run(
            EXPERIMENT,
            run_id,
            meta,
            epochs=epochs,
            config=cfg,
            validation=space.validate_config(cfg),
        )

    persist(
        base,
        "run-01",
        metrics={**DEFAULT_METRICS, "accuracy": 0.80},
        start_time="2026-01-01T00:00:01Z",
    )
    persist(
        replace(base, activation="sigmoid"),
        "run-02",
        state="COMPLETED_WITH_WARNINGS",
        metrics={**DEFAULT_METRICS, "accuracy": 0.70},
        warnings=["He/Kaiming initialization is not conventionally recommended for sigmoid"],
        start_time="2026-01-01T00:00:02Z",
    )
    persist(
        replace(base, max_epochs=5),
        "run-03",
        metrics={**DEFAULT_METRICS, "accuracy": 0.85},
        start_time="2026-01-01T00:00:03Z",
    )
    persist(
        base,
        "run-04",
        state="FAILED",
        failure="NaN detected in training loss",
        start_time="2026-01-01T00:00:04Z",
    )
    persist(
        replace(base, seed=7),
        "run-05",
        metrics={**DEFAULT_METRICS, "accuracy": 0.70},
        start_time="2026-01-01T00:00:05Z",
    )
    persist(
        replace(base, seed=13),
        "run-06",
        metrics={**DEFAULT_METRICS, "accuracy": 0.90},
        start_time="2026-01-01T00:00:06Z",
    )
    persist(
        replace(base, preprocessing=replace(base.preprocessing, class_weight="balanced")),
        "run-07",
        metrics={**DEFAULT_METRICS, "accuracy": 0.60},
        start_time="2026-01-01T00:00:07Z",
    )

    results.append_decision(
        EXPERIMENT,
        decision_type="label",
        justification="inspect training stability",
        actor="analyst",
        run_id="run-02",
        label="PENDING_REVIEW",
    )
    return results


@pytest.fixture
def exported(tmp_path: Path):
    results = _build_store(tmp_path / "results")
    out_dir = tmp_path / "exports"
    result = export.export_all(results, EXPERIMENT, out_dir, baseline_run_id="run-01")
    return results, result, out_dir / EXPERIMENT


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_run(rows: list[dict], run_id: str) -> dict:
    return next(row for row in rows if row["run_id"] == run_id)


# ---------------------------------------------------------------------------
# files / naming / organization
# ---------------------------------------------------------------------------


def test_export_all_creates_all_tables(exported):
    _, result, export_dir = exported
    expected = {
        "master",
        "baseline_comparison",
        "aggregates",
        "epochs",
        "configs",
        "decisions",
        "warnings_summary",
    }
    assert set(result.files) == expected
    for name, path in result.files.items():
        assert Path(path).exists()
        assert Path(path).name.startswith(f"{name}_")
        assert Path(path).suffix == ".csv"
    assert result.export_dir == export_dir
    assert result.timestamp in result.files["master"].name


def test_export_result_to_dict(exported):
    _, result, _ = exported
    payload = result.to_dict()
    assert payload["experiment_id"] == EXPERIMENT
    assert payload["baseline_run_id"] == "run-01"
    assert set(payload["files"]) == set(result.files)


# ---------------------------------------------------------------------------
# master table
# ---------------------------------------------------------------------------


def test_master_includes_all_runs_and_required_columns(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("master_*.csv")).name)
    assert len(rows) == 7
    header = set(rows[0])
    required = {
        "run_id",
        "config_id",
        "space_key",
        "dataset",
        "architecture",
        "activation",
        "optimizer",
        "lr",
        "init",
        "batch_size",
        "seed",
        "split_train",
        "split_val",
        "split_test",
        "max_epochs",
        "epochs_executed",
        "execution_state",
        "start_time",
        "end_time",
        "device",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "loss",
        "val_loss",
        "val_accuracy",
        "warnings",
        "failure_reason",
        "training_time",
        "n_params",
    }
    assert required <= header


def test_master_failed_run_present_with_reason_and_na_metrics(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("master_*.csv")).name)
    failed = _by_run(rows, "run-04")
    assert failed["execution_state"] == "FAILED"
    assert failed["failure_reason"]
    assert failed["accuracy"] == export.NA_MARKER
    assert failed["loss"] == export.NA_MARKER
    # never a fabricated zero
    assert failed["accuracy"] != "0"
    assert failed["accuracy"] != "0.0"


def test_master_completed_run_keeps_metrics_and_warnings(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("master_*.csv")).name)
    completed = _by_run(rows, "run-01")
    assert float(completed["accuracy"]) == pytest.approx(0.80)
    assert completed["epochs_executed"] == "2"
    warned = _by_run(rows, "run-02")
    assert warned["execution_state"] == "COMPLETED_WITH_WARNINGS"
    assert warned["warnings"]
    assert warned["space_key"]


# ---------------------------------------------------------------------------
# baseline comparison
# ---------------------------------------------------------------------------


def test_comparison_identifies_baseline_and_deltas(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("baseline_comparison_*.csv")).name)
    assert rows
    assert all(row["baseline_run_id"] == "run-01" for row in rows)
    assert all(row["baseline_config_id"] for row in rows)
    assert all(row["baseline_rationale"] for row in rows)

    controlled = _by_run(rows, "run-02")
    assert controlled["comparison_type"] == "controlled"
    assert controlled["comparability_flag"] == "comparable"
    assert float(controlled["delta_accuracy"]) == pytest.approx(-0.10, abs=1e-9)


def test_comparison_non_comparable_deltas_are_na(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("baseline_comparison_*.csv")).name)
    non_comparable = _by_run(rows, "run-03")
    assert non_comparable["comparability_flag"] == "non_comparable"
    assert non_comparable["delta_accuracy"] == export.NA_MARKER
    assert non_comparable["delta_loss"] == export.NA_MARKER
    assert non_comparable["warnings"]
    # never fabricated zeros
    assert non_comparable["delta_accuracy"] != "0.0"

    failed = _by_run(rows, "run-04")
    assert failed["state"] == "FAILED"
    assert failed["failure_reason"]
    assert failed["delta_accuracy"] == export.NA_MARKER


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------


def test_aggregates_exclude_failed_and_report_statistics(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("aggregates_*.csv")).name)
    base_group = next(row for row in rows if row["n_total"] == "4")
    assert base_group["n_valid"] == "3"
    assert base_group["n_failed"] == "1"
    assert base_group["insufficient_repetitions"] == "False"
    assert float(base_group["accuracy_mean"]) == pytest.approx(0.80, abs=1e-9)
    assert float(base_group["accuracy_std"]) == pytest.approx(0.10, abs=1e-9)
    assert float(base_group["failed_proportion"]) == pytest.approx(0.25, abs=1e-9)
    # failed run never contributes to the metric count
    assert base_group["accuracy_n"] == "3"


def test_aggregates_flag_insufficient_repetitions(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("aggregates_*.csv")).name)
    single = next(row for row in rows if row["n_total"] == "1")
    assert single["insufficient_repetitions"] == "True"


def test_aggregate_columns_cover_record_fields(exported):
    st, _, _ = exported
    runs = st.query_runs(EXPERIMENT)
    records = aggregate.aggregate_records(aggregate.aggregate_runs(runs))
    for record in records:
        assert set(record) <= set(export.AGGREGATE_COLUMNS)


# ---------------------------------------------------------------------------
# epochs
# ---------------------------------------------------------------------------


def test_epochs_exported_ordered_and_filterable(tmp_path):
    results = _build_store(tmp_path / "results")
    full = export.export_all(results, EXPERIMENT, tmp_path / "exports", baseline_run_id="run-01")
    rows = _read(full.files["epochs"])
    assert set(rows[0]) == set(store.EPOCH_COLUMNS)
    assert len(rows) == 12  # 6 completed runs x 2 epochs (run-04 FAILED has none)
    keys = [(row["run_id"], int(row["epoch"])) for row in rows]
    assert keys == sorted(keys)

    filtered = export.export_all(
        results,
        EXPERIMENT,
        tmp_path / "exports-filtered",
        baseline_run_id="run-01",
        epoch_run_ids=["run-01"],
        include_plots=False,
    )
    filtered_rows = _read(filtered.files["epochs"])
    assert {row["run_id"] for row in filtered_rows} == {"run-01"}
    assert len(filtered_rows) == 2


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def test_configs_export_includes_reproducibility_and_validation(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("configs_*.csv")).name)
    assert len(rows) == 6
    balanced = next(row for row in rows if row["class_weight"] == "balanced")
    assert balanced["scaler"] == "standard"
    assert balanced["convergence_enabled"] == "False"
    assert balanced["validation_state"] in {"VALID", "WARNING"}

    for row in rows:
        assert row["config_json"]
        restored = json.loads(row["config_json"])
        assert identity.compute_config_id(ExperimentConfig.from_dict(restored)) == row["config_id"]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def test_decisions_export_in_append_order(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("decisions_*.csv")).name)
    assert set(rows[0]) == set(export.DECISION_COLUMNS)
    # pending-review entry was appended before export, baseline decision during export
    pending = next(row for row in rows if row["label"] == "PENDING_REVIEW")
    assert pending["run_id"] == "run-02"
    assert pending["justification"] == "inspect training stability"
    assert any(row["decision_type"] == "validate" and row["actor"] == "export" for row in rows)


# ---------------------------------------------------------------------------
# warnings / pending review summary
# ---------------------------------------------------------------------------


def test_warnings_summary_lists_warnings_failure_and_pending(exported):
    _, _, export_dir = exported
    rows = _read(export_dir / next(export_dir.glob("warnings_summary_*.csv")).name)
    categories = {row["category"] for row in rows}
    assert {"warning", "failure", "pending_review"} <= categories
    assert any(row["run_id"] == "run-02" and row["category"] == "warning" for row in rows)
    assert any(row["run_id"] == "run-04" and row["category"] == "failure" for row in rows)
    assert any(
        row["category"] == "pending_review" and "stability" in row["message"] for row in rows
    )


# ---------------------------------------------------------------------------
# baseline persistence (deferred from PR4)
# ---------------------------------------------------------------------------


def test_baseline_persisted_to_metadata_and_decision_once(tmp_path):
    results = _build_store(tmp_path / "results")
    out_dir = tmp_path / "exports"
    export.export_all(results, EXPERIMENT, out_dir, baseline_run_id="run-01")
    stored = results.load_experiment_metadata(EXPERIMENT)
    assert stored["baseline"]["run_id"] == "run-01"
    assert stored["baseline"]["rationale"]

    baseline_decisions = [
        d
        for d in results.load_decisions(EXPERIMENT)
        if d.get("actor") == "export" and d.get("decision_type") == "validate"
    ]
    assert len(baseline_decisions) == 1

    # a second export reuses the stored baseline and does not duplicate the entry
    export.export_all(results, EXPERIMENT, out_dir, include_plots=False)
    baseline_decisions = [
        d
        for d in results.load_decisions(EXPERIMENT)
        if d.get("actor") == "export" and d.get("decision_type") == "validate"
    ]
    assert len(baseline_decisions) == 1


def test_export_without_baseline_still_succeeds(tmp_path):
    results = _build_store(tmp_path / "results")
    result = export.export_all(results, EXPERIMENT, tmp_path / "exports", include_plots=False)
    assert result.baseline_run_id is None
    assert Path(result.files["baseline_comparison"]).exists()
    assert any("baseline not selected" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# optional plots are non-blocking
# ---------------------------------------------------------------------------


def test_plot_failure_is_non_blocking(tmp_path, monkeypatch):
    results = _build_store(tmp_path / "results")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no display available")

    monkeypatch.setattr(export, "_write_plots", _boom)
    result = export.export_all(results, EXPERIMENT, tmp_path / "exports", baseline_run_id="run-01")
    assert Path(result.files["master"]).exists()
    assert any("plot generation skipped" in warning for warning in result.warnings)


def test_invalid_grouping_rejected(tmp_path):
    results = _build_store(tmp_path / "results")
    with pytest.raises(ValueError):
        export.export_all(results, EXPERIMENT, tmp_path / "exports", grouping="bogus")


def test_na_marker_never_zero():
    assert export.NA_MARKER == "N/A"
