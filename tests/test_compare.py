"""Tests for the baseline comparator (AC7, AC8, AC10, AC12)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from prebenchmark.compare import (
    DELTA_METRICS,
    RELATIVE_BASIS,
    BaselineSelection,
    ComparisonTable,
    check_comparability,
    compare_to_baseline,
    documented_baseline_config,
    filter_runs,
    select_baseline,
)
from prebenchmark.config import (
    ArchitectureSpec,
    ConvergenceSpec,
    DatasetSpec,
    ExperimentConfig,
)
from prebenchmark.identity import compute_config_id

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
    """Runner-shaped metadata record: config_snapshot + execution_state + test metrics."""
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
        metadata.pop("execution_state")
        metadata["state"] = "FAILED"
        metadata["failure"] = {"reason": "exploded: NaN in training loss", "traceback": "..."}
        metadata["final_metrics"] = {}
        metadata.pop("training_time")
    metadata.update(overrides)
    return metadata


BASELINE_RUN = make_run(BASE, "run-baseline")


def run_ids(records: list[dict]) -> list[str]:
    return [r["run_id"] for r in records]


# ---------------------------------------------------------------------------
# Baseline selection (explicit, recorded, never silent)
# ---------------------------------------------------------------------------


def test_select_baseline_explicit_run_id() -> None:
    cand = make_run(replace(BASE, lr=0.01), "run-cand")
    baseline = select_baseline(
        [BASELINE_RUN, cand], baseline_run_id="run-baseline", rationale="reference run"
    )
    assert isinstance(baseline, BaselineSelection)
    assert baseline.run_id == "run-baseline"
    assert baseline.config_id == compute_config_id(BASE)
    assert baseline.state == "COMPLETED"
    assert baseline.rationale == "reference run"
    assert baseline.timestamp  # selection timestamp recorded


def test_select_baseline_by_config_id_picks_earliest() -> None:
    later = make_run(BASE, "run-later", start_time="2026-01-02T00:00:00Z")
    baseline = select_baseline([later, BASELINE_RUN], baseline_config_id=compute_config_id(BASE))
    assert baseline.run_id == "run-baseline"
    assert "earliest" in baseline.rationale


def test_select_baseline_documented_default_is_explicit() -> None:
    baseline = select_baseline([BASELINE_RUN], use_documented_default=True)
    assert baseline.run_id == "run-baseline"
    assert baseline.config_id == compute_config_id(documented_baseline_config())
    assert "documented default" in baseline.rationale


def test_select_baseline_documented_default_matches_across_dataset_and_protocol() -> None:
    """La línea base documentada se define por sus dimensiones, no por el dataset ni el protocolo."""
    cfg = replace(
        BASE,
        dataset=DatasetSpec(
            name="satimage",
            source="OpenML satimage v1",
            target="target",
            n_features=36,
        ),
        architecture=ArchitectureSpec(
            name="satimage_mlp",
            input_dim=36,
            hidden_layers=[32, 16],
            output_dim=6,
        ),
        convergence=ConvergenceSpec(
            enabled=True,
            metric="val_loss",
            patience=20,
            min_delta=1e-4,
            mode="min",
        ),
    )
    run = make_run(cfg, "run-satimage")
    baseline = select_baseline([run], use_documented_default=True)
    assert baseline.run_id == "run-satimage"
    assert baseline.config_id == compute_config_id(cfg)
    assert "dataset: satimage" in baseline.rationale


def test_select_baseline_requires_explicit_choice() -> None:
    with pytest.raises(ValueError, match="explicit"):
        select_baseline([BASELINE_RUN])
    with pytest.raises(ValueError, match="exactly one"):
        select_baseline(
            [BASELINE_RUN],
            baseline_run_id="run-baseline",
            baseline_config_id=compute_config_id(BASE),
        )
    with pytest.raises(ValueError, match="not found"):
        select_baseline([BASELINE_RUN], baseline_run_id="nope")
    with pytest.raises(ValueError, match="no run found"):
        select_baseline([BASELINE_RUN], baseline_config_id="cfg-v1-000000000000")
    with pytest.raises(ValueError, match="no run found"):
        # documented default asked for, but no default-config run present
        select_baseline([make_run(replace(BASE, lr=0.01), "run-cand")], use_documented_default=True)


def test_select_baseline_rejects_failed_and_empty_rationale() -> None:
    failed = make_run(BASE, "run-fail", state="FAILED")
    with pytest.raises(ValueError, match="FAILED"):
        select_baseline([failed], baseline_run_id="run-fail")
    with pytest.raises(ValueError, match="non-empty"):
        select_baseline([BASELINE_RUN], baseline_run_id="run-baseline", rationale="   ")


# ---------------------------------------------------------------------------
# Delta correctness (controlled comparison)
# ---------------------------------------------------------------------------


def _controlled_candidate() -> dict:
    return make_run(
        replace(BASE, lr=0.01),
        "run-cand",
        train_time=8.5,
        metrics={
            "accuracy": 0.62,
            "balanced_accuracy": 0.57,
            "precision_macro": 0.53,
            "recall_macro": 0.55,
            "f1_macro": 0.54,
            "loss": 1.10,
        },
    )


def test_controlled_comparison_computes_correct_deltas() -> None:
    cand = _controlled_candidate()
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]

    assert row.comparison_type == "controlled"
    assert row.varied_dimensions == ("lr",)
    assert row.informative_only is False
    assert row.comparable is True
    assert row.evaluation_set == "test"

    # absolute deltas: candidate - baseline on the SAME evaluation set (test)
    assert row.deltas["accuracy"] == pytest.approx(0.02)
    assert row.deltas["balanced_accuracy"] == pytest.approx(0.02)
    assert row.deltas["precision_macro"] == pytest.approx(0.03)
    assert row.deltas["recall_macro"] == pytest.approx(0.03)
    assert row.deltas["f1_macro"] == pytest.approx(0.03)
    assert row.deltas["loss"] == pytest.approx(-0.10)
    assert row.deltas["training_time"] == pytest.approx(-1.5)

    # raw metric profile preserved alongside the deltas
    assert row.metrics["accuracy"] == pytest.approx(0.62)
    assert row.metrics["training_time"] == pytest.approx(8.5)


def test_relative_deltas_have_stated_basis() -> None:
    cand = _controlled_candidate()
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]

    assert row.relative_deltas["accuracy"] == pytest.approx(0.02 / 0.60)
    assert row.relative_deltas["loss"] == pytest.approx(-0.10 / 1.20)
    assert row.relative_deltas["training_time"] == pytest.approx(-1.5 / 10.0)
    assert row.relative_basis == RELATIVE_BASIS


def test_relative_delta_na_when_denominator_uninterpretable() -> None:
    zero_base = make_run(BASE, "run-zero", metrics={"accuracy": 0.0})
    cand = make_run(BASE, "run-cand", metrics={"accuracy": 0.5})
    row = compare_to_baseline(zero_base, [cand]).rows[0]
    assert row.deltas["accuracy"] == pytest.approx(0.5)  # absolute still valid
    assert row.relative_deltas["accuracy"] is None  # denominator zero -> N/A


def test_delta_na_when_metric_missing_in_candidate() -> None:
    cand = make_run(replace(BASE, lr=0.01), "run-cand", metrics={"loss": None})
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]
    assert row.deltas["loss"] is None
    assert "loss" not in row.metrics


def test_relative_deltas_can_be_omitted() -> None:
    cand = _controlled_candidate()
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    table = compare_to_baseline(baseline, [cand], include_relative=False)
    row = table.rows[0]
    assert all(value is None for value in row.relative_deltas.values())
    assert row.relative_basis == ""
    record = table.to_records(include_relative=False)[0]
    for metric in DELTA_METRICS:
        assert f"rel_{metric}" not in record


# ---------------------------------------------------------------------------
# Controlled vs multi-variable vs identical
# ---------------------------------------------------------------------------


def test_multi_variable_comparison_flagged_informative_only() -> None:
    cand = make_run(replace(BASE, lr=0.01, activation="tanh"), "run-multi")
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]

    assert row.comparison_type == "multi_variable"
    assert row.varied_dimensions == ("activation", "lr")
    assert row.informative_only is True
    assert row.comparable is True  # protocol identical -> deltas shown, cause not attributable
    assert row.deltas["accuracy"] is not None

    record = row_to_record(row)
    assert record["informative_only"] is True
    assert record["controlled_variable"] == ""


def row_to_record(row) -> dict:
    return ComparisonTable(
        baseline_run_id="b",
        baseline_config_id="c",
        baseline_state="COMPLETED",
        baseline_rationale="",
        baseline_timestamp="",
        evaluation_set="test",
        baseline_metrics={},
        rows=(row,),
    ).to_records()[0]


def test_identical_run_shows_zero_deltas() -> None:
    twin = make_run(BASE, "run-twin")
    baseline = select_baseline([BASELINE_RUN, twin], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [twin]).rows[0]
    assert row.comparison_type == "identical"
    assert row.informative_only is False
    assert row.deltas["accuracy"] == pytest.approx(0.0)
    assert row.deltas["training_time"] == pytest.approx(0.0)


def test_class_weight_difference_is_a_controlled_comparison() -> None:
    """Balancear clases es una variable controlada, no una ejecución idéntica.

    Dos ejecuciones que solo difieren en ``class_weight`` mantienen el mismo
    dataset, arquitectura, partición y protocolo: la comparación es válida (una
    sola variable) y debe quedar etiquetada como tal.
    """
    balanced_cfg = replace(BASE, preprocessing=replace(BASE.preprocessing, class_weight="balanced"))
    cand = make_run(balanced_cfg, "run-balanced")
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]

    assert row.comparable is True
    assert row.comparison_type == "controlled"
    assert row.varied_dimensions == ("class_weight",)
    assert row.informative_only is False
    assert row.deltas["balanced_accuracy"] == pytest.approx(0.0)

    record = row_to_record(row)
    assert record["controlled_variable"] == "class_weight"


def test_class_weight_plus_space_difference_is_multi_variable() -> None:
    balanced_cfg = replace(
        BASE,
        lr=0.01,
        preprocessing=replace(BASE.preprocessing, class_weight="balanced"),
    )
    cand = make_run(balanced_cfg, "run-balanced-lr")
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]

    assert row.comparison_type == "multi_variable"
    assert row.varied_dimensions == ("lr", "class_weight")
    assert row.informative_only is True


# ---------------------------------------------------------------------------
# Comparability guards -> refuse, never fake deltas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg_overrides, expected_reason",
    [
        (
            {"dataset": replace(BASE.dataset, name="other_dataset")},
            "datasets differ",
        ),
        (
            {"architecture": replace(BASE.architecture, hidden_layers=[64, 16])},
            "architectures differ",
        ),
        (
            {"split": replace(BASE.split, train=0.5, val=0.3, test=0.2)},
            "split proportions",
        ),
        (
            {"max_epochs": 250},
            "max_epochs differ",
        ),
        (
            {"run_type": "benchmark"},
            "run_type differs",
        ),
        (
            {
                "convergence": replace(
                    BASE.convergence,
                    enabled=True,
                    metric="val_loss",
                    patience=10,
                    min_delta=0.001,
                )
            },
            "convergence settings differ",
        ),
    ],
    ids=["dataset", "architecture", "split", "max_epochs", "run_type", "convergence"],
)
def test_guard_flags_non_comparable(cfg_overrides: dict, expected_reason: str) -> None:
    cand = make_run(replace(BASE, **cfg_overrides), "run-other")

    result = check_comparability(BASELINE_RUN, cand)
    assert result.comparable is False
    assert any(expected_reason in reason for reason in result.reasons)

    baseline = select_baseline([BASELINE_RUN], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [cand]).rows[0]
    assert row.comparison_type == "non_comparable"
    assert row.informative_only is True
    assert all(delta is None for delta in row.deltas.values())  # N/A, never faked
    assert row.metrics["accuracy"] == pytest.approx(0.60)  # raw profile still visible


def test_eval_set_mismatch_and_unlabelled_refused() -> None:
    on_val = make_run(BASE, "run-val", metrics={"evaluation_set": "val", "accuracy": 0.61})
    result = check_comparability(BASELINE_RUN, on_val)
    assert result.comparable is False
    assert any("evaluation sets differ" in reason for reason in result.reasons)

    unlabelled = make_run(BASE, "run-nolabel")
    unlabelled["final_metrics"] = {"accuracy": 0.61}
    result2 = check_comparability(BASELINE_RUN, unlabelled)
    assert result2.comparable is False
    assert any("not labelled" in reason for reason in result2.reasons)


# ---------------------------------------------------------------------------
# Failed runs: listed, reason shown, no deltas
# ---------------------------------------------------------------------------


def test_failed_candidate_listed_with_reason_and_no_deltas() -> None:
    failed = make_run(BASE, "run-fail", state="FAILED")
    baseline = select_baseline([BASELINE_RUN, failed], baseline_run_id="run-baseline")
    row = compare_to_baseline(baseline, [failed]).rows[0]

    assert row.state == "FAILED"
    assert row.comparable is False
    assert row.comparison_type == "non_comparable"
    assert row.failure_reason == "exploded: NaN in training loss"
    assert row.metrics == {}
    assert all(delta is None for delta in row.deltas.values())

    record = row_to_record(row)
    assert record["state"] == "FAILED"
    assert "exploded" in record["failure_reason"]
    assert record["comparability_flag"] == "non_comparable"


def test_failed_baseline_refuses_comparison() -> None:
    failed = make_run(BASE, "run-fail", state="FAILED")
    with pytest.raises(ValueError, match="FAILED"):
        select_baseline([failed], baseline_run_id="run-fail")
    with pytest.raises(ValueError, match="FAILED"):
        compare_to_baseline(failed, [BASELINE_RUN])


def test_compare_to_baseline_accepts_raw_metadata() -> None:
    cand = _controlled_candidate()
    table = compare_to_baseline(BASELINE_RUN, [cand])
    assert table.baseline_run_id == "run-baseline"
    assert "raw metadata" in table.baseline_rationale
    assert table.rows[0].comparison_type == "controlled"


# ---------------------------------------------------------------------------
# No composite score; full profile preserved; structured table
# ---------------------------------------------------------------------------


def test_no_composite_score_and_full_profile_preserved() -> None:
    cand = _controlled_candidate()
    baseline = select_baseline([BASELINE_RUN, cand], baseline_run_id="run-baseline")
    table = compare_to_baseline(baseline, [cand])

    for record in table.to_records():
        for key in record:
            assert "composite" not in key.lower()
            assert "weighted" not in key.lower()

    # the absolute-delta schema is stable: every metric has a delta column (N/A as None)
    assert {f"delta_{m}" for m in DELTA_METRICS} <= set(table.to_records()[0])
    # raw baseline profile kept individually
    assert table.baseline_metrics["accuracy"] == pytest.approx(0.60)
    assert table.baseline_metrics["training_time"] == pytest.approx(10.0)
    assert table.evaluation_set == "test"


def test_comparison_table_preserves_candidate_order() -> None:
    cand_a = make_run(replace(BASE, lr=0.01), "run-a")
    cand_b = make_run(replace(BASE, lr=0.1), "run-b")
    baseline = select_baseline([BASELINE_RUN, cand_a, cand_b], baseline_run_id="run-baseline")
    rows = compare_to_baseline(baseline, [cand_a, cand_b]).rows
    assert [row.run_id for row in rows] == ["run-a", "run-b"]  # no hidden ranking


# ---------------------------------------------------------------------------
# Comparison filters
# ---------------------------------------------------------------------------


def test_filter_runs_by_dimensions_and_dataset() -> None:
    runs = [
        make_run(BASE, "r1"),
        make_run(replace(BASE, lr=0.01), "r2"),
        make_run(replace(BASE, activation="tanh"), "r3", state="FAILED"),
        make_run(replace(BASE, dataset=replace(BASE.dataset, name="other_dataset")), "r4"),
        make_run(replace(BASE, optimizer="sgd", seed=7, batch_size=64), "r5"),
    ]
    assert run_ids(filter_runs(runs, activation="relu")) == ["r1", "r2", "r4", "r5"]
    assert run_ids(filter_runs(runs, lr=0.01)) == ["r2"]
    assert run_ids(filter_runs(runs, dataset="other_dataset")) == ["r4"]
    assert run_ids(filter_runs(runs, architecture="wine_quality_mlp")) == run_ids(runs)
    assert run_ids(filter_runs(runs, optimizer="sgd")) == ["r5"]
    assert run_ids(filter_runs(runs, init="he")) == run_ids(runs)
    assert run_ids(filter_runs(runs, batch_size=64)) == ["r5"]
    assert run_ids(filter_runs(runs, seed=7)) == ["r5"]
    assert run_ids(filter_runs(runs, run_type="prebenchmark")) == run_ids(runs)
    # no filters -> full set (copies)
    assert [r["run_id"] for r in filter_runs(runs)] == run_ids(runs)


def test_filter_runs_status_alias_and_conflict() -> None:
    runs = [make_run(BASE, "r1"), make_run(BASE, "r2", state="FAILED")]
    assert run_ids(filter_runs(runs, status="FAILED")) == ["r2"]
    assert run_ids(filter_runs(runs, state="FAILED")) == ["r2"]
    assert run_ids(filter_runs(runs, status="COMPLETED")) == ["r1"]
    with pytest.raises(ValueError, match="either 'status' or 'state'"):
        filter_runs(runs, status="FAILED", state="FAILED")
