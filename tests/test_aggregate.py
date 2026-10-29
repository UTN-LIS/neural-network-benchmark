"""Tests for repetition aggregation (AC9, AC10, AC12).

Covers statistics correctness (including sample ``ddof=1``), the
insufficient-repetitions flag, failed/unstable accounting, multi-seed grouping,
the 36-space context, and CSV-ready flattening.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from prebenchmark.aggregate import (
    AGGREGATE_METRICS,
    DEFAULT_MIN_REPETITIONS,
    RunGroup,
    aggregate_group,
    aggregate_records,
    aggregate_runs,
    compute_group_key,
    group_runs,
)
from prebenchmark.config import ExperimentConfig, PreprocessingSpec
from prebenchmark.identity import compute_config_id
from prebenchmark.space import compute_space_key

BASE = ExperimentConfig()


def make_run(
    cfg: ExperimentConfig,
    run_id: str,
    *,
    state: str = "COMPLETED",
    train_time: float = 10.0,
    metrics: dict | None = None,
    start_time: str = "2026-01-01T00:00:00Z",
    **overrides: object,
) -> dict:
    """Runner-shaped metadata record: config_snapshot + execution_state + metrics."""
    full_metrics: dict = {
        "evaluation_set": "test",
        "accuracy": 0.60,
        "balanced_accuracy": 0.55,
        "precision_macro": 0.50,
        "recall_macro": 0.52,
        "f1_macro": 0.51,
        "confusion_matrix": [],
        "loss": 1.20,
    }
    if metrics:
        full_metrics.update(metrics)
    metadata: dict = {
        "run_id": run_id,
        "config_id": compute_config_id(cfg),
        "run_type": cfg.run_type,
        "start_time": start_time,
        "end_time": "2026-01-01T00:01:00Z",
        "config_snapshot": cfg.to_dict(),
        "dataset": cfg.dataset.to_dict(),
        "architecture": cfg.architecture.to_dict(),
        "activation": cfg.activation,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "init": cfg.init,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "seed": cfg.seed,
        "split": cfg.split.to_dict(),
        "convergence": cfg.convergence.to_dict(),
        "execution_state": state,
        "final_metrics": full_metrics,
        "training_time": train_time,
    }
    if state == "FAILED":
        # Runner writes ``state`` (not ``execution_state``) on failure.
        metadata.pop("execution_state")
        metadata["state"] = "FAILED"
        metadata["failure"] = {"reason": "exploded: NaN in training loss", "traceback": "..."}
        metadata["final_metrics"] = {}
        metadata.pop("training_time")
    metadata.update(overrides)
    return metadata


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_group_runs_groups_same_configuration() -> None:
    runs = [make_run(BASE, "r1"), make_run(BASE, "r2"), make_run(BASE, "r3")]
    groups = group_runs(runs)
    assert len(groups) == 1
    assert groups[0].size == 3
    assert groups[0].config_id == compute_config_id(BASE)


def test_group_runs_distinguishes_protocol_differences() -> None:
    runs = [
        make_run(BASE, "base"),
        make_run(replace(BASE, max_epochs=250), "epochs"),
        make_run(replace(BASE, lr=0.01), "lr"),
    ]
    groups = group_runs(runs)
    assert len(groups) == 3  # differing protocol/space settings stay separated


def test_group_runs_distinguishes_class_weight() -> None:
    balanced = replace(
        BASE, preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced")
    )
    runs = [make_run(BASE, "plain"), make_run(balanced, "balanced")]
    # class_weight is outside space_key but inside the full configuration, so it
    # is a protocol difference and must split the group.
    assert len(group_runs(runs)) == 2


def test_group_runs_multi_seed_same_protocol_grouped() -> None:
    runs = [make_run(replace(BASE, seed=seed), f"run-{seed}") for seed in (42, 7, 13)]
    groups = group_runs(runs)
    assert len(groups) == 1  # different seeds share the repetition group
    group = groups[0]
    assert group.size == 3
    # config_id is seed-sensitive, so the members have distinct config ids...
    assert len(group.config_ids) == 3
    # ...but every seed is preserved in the member list.
    aggregated = aggregate_group(group)
    assert aggregated.seeds == (7, 13, 42)


def test_group_runs_strict_config_id_mode_splits_seeds() -> None:
    runs = [make_run(replace(BASE, seed=seed), f"run-{seed}") for seed in (42, 7)]
    strict = group_runs(runs, grouping="config_id")
    assert len(strict) == 2  # strict, seed-sensitive identity


def test_group_runs_records_space_key_context() -> None:
    group = group_runs([make_run(BASE, "r1")])[0]
    assert group.space_key == compute_space_key(BASE)
    assert aggregate_group(group).space_key == compute_space_key(BASE)


def test_group_runs_reads_explicit_space_key_when_present() -> None:
    meta = make_run(BASE, "r1", space_key="space-v1-explicit")
    group = group_runs([meta])[0]
    assert group.space_key == "space-v1-explicit"


def test_group_runs_records_snapshot_and_creation_time() -> None:
    group = group_runs([make_run(BASE, "r1")], created_at="2026-05-05T00:00:00Z")[0]
    assert group.created_at == "2026-05-05T00:00:00Z"
    assert group.config_snapshot["activation"] == "relu"
    assert aggregate_group(group).created_at == "2026-05-05T00:00:00Z"


def test_group_key_rejects_unknown_grouping_mode() -> None:
    with pytest.raises(ValueError, match="grouping"):
        compute_group_key(make_run(BASE, "r1"), grouping="nope")


# ---------------------------------------------------------------------------
# Statistics correctness
# ---------------------------------------------------------------------------


def test_aggregate_group_basic_statistics_with_ddof_one() -> None:
    runs = [
        make_run(BASE, "r1", metrics={"accuracy": 0.6}),
        make_run(BASE, "r2", metrics={"accuracy": 0.7}),
        make_run(BASE, "r3", metrics={"accuracy": 0.8}),
    ]
    aggregated = aggregate_runs(runs)[0]

    assert aggregated.n_total == 3
    assert aggregated.n_valid == 3
    accuracy = aggregated.metrics["accuracy"]
    assert accuracy.n == 3
    assert accuracy.mean == pytest.approx(0.7)
    # sample standard deviation (ddof=1): sqrt(0.02 / 2) == 0.1 exactly
    assert accuracy.std == pytest.approx(0.1)
    assert accuracy.minimum == pytest.approx(0.6)
    assert accuracy.maximum == pytest.approx(0.8)
    assert accuracy.spread == pytest.approx(0.2)
    assert accuracy.sufficient is True


def test_aggregate_includes_best_epoch_validation_metrics() -> None:
    runs = [
        make_run(BASE, "r1", validation_metrics={"evaluation_set": "validation", "epoch": 3, "loss": 0.30, "accuracy": 0.70}),
        make_run(BASE, "r2", validation_metrics={"evaluation_set": "validation", "epoch": 4, "loss": 0.20, "accuracy": 0.80}),
        make_run(BASE, "r3", validation_metrics={"evaluation_set": "validation", "epoch": 2, "loss": 0.10, "accuracy": 0.90}),
    ]
    aggregated = aggregate_runs(runs)[0]

    val_loss = aggregated.metrics["val_loss"]
    assert val_loss.n == 3
    assert val_loss.mean == pytest.approx(0.20)
    assert val_loss.minimum == pytest.approx(0.10)
    assert val_loss.maximum == pytest.approx(0.30)

    val_acc = aggregated.metrics["val_accuracy"]
    assert val_acc.n == 3
    assert val_acc.mean == pytest.approx(0.80)
    # both evaluation sets are disclosed (test for final, validation for the best epoch)
    assert set(aggregated.evaluation_sets) == {"test", "validation"}


def test_aggregate_validation_metrics_missing_yields_zero_valid() -> None:
    # Legacy/test records without ``validation_metrics`` contribute no valid values
    # for the validation metrics (never a fabricated zero).
    aggregated = aggregate_runs([make_run(BASE, "r1")])[0]
    assert aggregated.metrics["val_loss"].n == 0
    assert aggregated.metrics["val_loss"].mean is None
    assert aggregated.metrics["val_accuracy"].n == 0


def test_std_is_none_but_mean_reported_for_single_run() -> None:
    aggregated = aggregate_runs([make_run(BASE, "r1", metrics={"accuracy": 0.6})])[0]
    accuracy = aggregated.metrics["accuracy"]
    assert accuracy.n == 1
    assert accuracy.mean == pytest.approx(0.6)
    assert accuracy.std is None  # ddof=1 undefined for n < 2


def test_all_aggregate_metrics_reported() -> None:
    aggregated = aggregate_runs([make_run(BASE, "r1")])[0]
    assert set(aggregated.metrics) == set(AGGREGATE_METRICS)


# ---------------------------------------------------------------------------
# Insufficient repetitions
# ---------------------------------------------------------------------------


def test_insufficient_repetitions_flag_and_recorded_threshold() -> None:
    runs = [make_run(BASE, "r1", metrics={"accuracy": 0.6}), make_run(BASE, "r2", metrics={"accuracy": 0.8})]
    aggregated = aggregate_runs(runs)[0]
    assert aggregated.n_valid == 2
    assert aggregated.min_repetitions == DEFAULT_MIN_REPETITIONS == 3
    assert aggregated.insufficient_repetitions is True
    # std is still reported, just flagged
    assert aggregated.metrics["accuracy"].std is not None
    assert aggregated.metrics["accuracy"].sufficient is False


def test_min_repetitions_is_configurable() -> None:
    runs = [make_run(BASE, "r1"), make_run(BASE, "r2")]
    aggregated = aggregate_runs(runs, min_repetitions=2)[0]
    assert aggregated.min_repetitions == 2
    assert aggregated.insufficient_repetitions is False
    assert aggregated.metrics["accuracy"].sufficient is True


def test_min_repetitions_must_be_positive() -> None:
    group = group_runs([make_run(BASE, "r1")])[0]
    with pytest.raises(ValueError, match="min_repetitions"):
        aggregate_group(group, min_repetitions=0)


# ---------------------------------------------------------------------------
# Failed / unstable accounting
# ---------------------------------------------------------------------------


def test_failed_runs_excluded_from_stats_but_counted() -> None:
    runs = [
        make_run(BASE, "ok1", metrics={"accuracy": 0.6}),
        make_run(BASE, "ok2", metrics={"accuracy": 0.8}),
        make_run(BASE, "bad", state="FAILED"),
    ]
    aggregated = aggregate_runs(runs)[0]

    assert aggregated.n_total == 3
    assert aggregated.n_valid == 2
    assert aggregated.n_failed == 1
    assert aggregated.failed_proportion == pytest.approx(1 / 3)
    assert aggregated.metrics["accuracy"].n == 2
    assert aggregated.metrics["accuracy"].mean == pytest.approx(0.7)

    failed_member = next(m for m in aggregated.members if m.run_id == "bad")
    assert failed_member.included is False
    assert failed_member.failure_reason == "exploded: NaN in training loss"
    assert aggregated.state_counts == {"COMPLETED": 2, "FAILED": 1}


def test_unstable_runs_counted_and_still_included_in_stats() -> None:
    runs = [
        make_run(BASE, "r1", metrics={"accuracy": 0.6}),
        make_run(
            BASE,
            "r2",
            state="COMPLETED_WITH_WARNINGS",
            exploratory_label="UNSTABLE",
            metrics={"accuracy": 0.8},
        ),
        make_run(BASE, "r3", state="UNSTABLE", metrics={"accuracy": 0.9}),
    ]
    aggregated = aggregate_runs(runs)[0]
    assert aggregated.n_valid == 3  # instability is not a failure
    assert aggregated.n_unstable == 2
    assert aggregated.unstable_proportion == pytest.approx(2 / 3)
    assert aggregated.metrics["accuracy"].n == 3


def test_members_preserve_run_and_seed_details() -> None:
    runs = [make_run(replace(BASE, seed=1), "r1"), make_run(replace(BASE, seed=2), "r2")]
    aggregated = aggregate_runs(runs)[0]
    assert {member.run_id for member in aggregated.members} == {"r1", "r2"}
    assert aggregated.seeds == (1, 2)
    assert all(member.included for member in aggregated.members)


# ---------------------------------------------------------------------------
# Per-metric validity (missing / non-finite values)
# ---------------------------------------------------------------------------


def test_per_metric_count_reduced_when_metric_missing() -> None:
    runs = [
        make_run(BASE, "r1"),
        make_run(BASE, "r2", metrics={"loss": None}),
        make_run(BASE, "r3"),
    ]
    aggregated = aggregate_runs(runs)[0]
    assert aggregated.metrics["accuracy"].n == 3
    assert aggregated.metrics["loss"].n == 2


def test_non_finite_metric_values_are_excluded() -> None:
    runs = [
        make_run(BASE, "r1"),
        make_run(BASE, "r2", metrics={"loss": float("nan")}),
        make_run(BASE, "r3"),
    ]
    aggregated = aggregate_runs(runs)[0]
    assert aggregated.metrics["loss"].n == 2


# ---------------------------------------------------------------------------
# Normalization reuse (runner- vs fixture-shaped records)
# ---------------------------------------------------------------------------


def test_aggregation_normalizes_state_and_execution_state() -> None:
    fixture = make_run(BASE, "fixture")
    fixture.pop("execution_state")
    fixture["state"] = "COMPLETED"  # store/fixture shape
    runs = [make_run(BASE, "runner"), fixture, make_run(BASE, "bad", state="FAILED")]
    aggregated = aggregate_runs(runs)[0]
    assert aggregated.n_valid == 2
    assert aggregated.n_failed == 1


# ---------------------------------------------------------------------------
# Empty guard / CSV readiness
# ---------------------------------------------------------------------------


def test_aggregate_group_rejects_empty_group() -> None:
    empty = RunGroup(
        group_key="k", config_id="c", config_ids=(), space_key=None,
        config_snapshot={}, runs=(), created_at="t",
    )
    with pytest.raises(ValueError, match="empty"):
        aggregate_group(empty)


def test_group_to_record_is_csv_ready_without_composite_score() -> None:
    aggregated = aggregate_runs([make_run(BASE, "r1", metrics={"accuracy": 0.6})])[0]
    record = aggregated.to_record()

    assert record["group_key"] == aggregated.group_key
    assert record["n_total"] == 1
    assert record["n_valid"] == 1
    assert record["insufficient_repetitions"] is True
    assert record["activation"] == "relu"
    assert record["optimizer"] == "adam"
    assert record["dataset"] == "wine_quality"
    assert record["batch_size"] == 32
    assert record["seeds"] == "42"
    assert record["run_ids"] == "r1"
    assert record["accuracy_n"] == 1
    assert record["accuracy_mean"] == pytest.approx(0.6)
    for key in record:
        assert "composite" not in key.lower()
        assert "weighted" not in key.lower()


def test_aggregate_records_flattens_all_groups() -> None:
    groups = aggregate_runs([make_run(BASE, "r1"), make_run(replace(BASE, lr=0.01), "r2")])
    records = aggregate_records(groups)
    assert len(records) == 2
    assert all("group_key" in record for record in records)
