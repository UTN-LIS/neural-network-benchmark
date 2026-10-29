"""Tests for the statistical analysis layer (``prebenchmark.analysis``).

The fixture writes real CSV files shaped like ``prebenchmark.export`` output, so
the tests exercise the same contract the CLI consumes: timestamped tables,
``N/A`` markers, ``config_json`` and per-epoch history. No training happens here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from prebenchmark import analysis


SEEDS = (42, 7, 123)
BASE_TIMESTAMP = "20260101T000000Z"

# (activation, optimizer, lr, init, class_weight) -> base F1
GROUPS = {
    ("relu", "adam", 0.01, "he", "none"): 0.80,
    ("relu", "adam", 0.01, "he", "balanced"): 0.70,
    ("tanh", "sgd", 0.001, "xavier", "none"): 0.60,
    ("tanh", "sgd", 0.001, "xavier", "balanced"): 0.50,
}


def _config(
    activation: str,
    optimizer: str,
    lr: float,
    init: str,
    class_weight: str,
    seed: int,
) -> dict:
    return {
        "activation": activation,
        "optimizer": optimizer,
        "lr": lr,
        "init": init,
        "seed": seed,
        "batch_size": 32,
        "max_epochs": 10,
        "loss": "cross_entropy",
        "run_type": "prebenchmark",
        "device": "cpu",
        "dataset": {"name": "wine_quality", "version": "sha256-test"},
        "architecture": {
            "name": "wine_quality_mlp",
            "input_dim": 11,
            "hidden_layers": [32, 16],
            "output_dim": 7,
        },
        "split": {"train": 0.6, "val": 0.2, "test": 0.2, "stratify": True},
        "preprocessing": {
            "scaler": "standard",
            "class_weight": None if class_weight == "none" else class_weight,
        },
        "convergence": {
            "enabled": True,
            "metric": "val_loss",
            "patience": 20,
            "min_delta": 1e-4,
            "mode": "min",
        },
    }


def _space_key(activation: str, optimizer: str, lr: float, init: str) -> str:
    return f"space-{activation}-{optimizer}-{lr}-{init}"


def _write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "N/A") for column in columns})


def write_export(
    directory: Path,
    *,
    timestamp: str = BASE_TIMESTAMP,
    f1_shift: float = 0.0,
    include_failed: bool = True,
) -> Path:
    """Write a complete synthetic export snapshot and return its directory."""
    master_rows: list[dict] = []
    config_rows: list[dict] = []
    epoch_rows: list[dict] = []
    run_counter = 0

    for (activation, optimizer, lr, init, class_weight), base_f1 in GROUPS.items():
        for seed_index, seed in enumerate(SEEDS):
            config = _config(activation, optimizer, lr, init, class_weight, seed)
            config_id = f"cfg-{activation}-{optimizer}-{lr}-{init}-{class_weight}-{seed}"
            run_id = f"20260101T{run_counter:06d}Z-{seed:06d}"
            run_counter += 1
            f1 = base_f1 + 0.01 * seed_index + f1_shift
            master_rows.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "space_key": _space_key(activation, optimizer, lr, init),
                    "experiment_id": directory.name,
                    "run_type": "prebenchmark",
                    "dataset": "wine_quality",
                    "dataset_version": "sha256-test",
                    "architecture": "wine_quality_mlp",
                    "activation": activation,
                    "optimizer": optimizer,
                    "lr": lr,
                    "init": init,
                    "loss_function": "cross_entropy",
                    "batch_size": 32,
                    "seed": seed,
                    "split_train": 0.6,
                    "split_val": 0.2,
                    "split_test": 0.2,
                    "split_stratify": True,
                    "max_epochs": 10,
                    "epochs_executed": 5,
                    "execution_state": "COMPLETED",
                    "start_time": "2026-01-01T00:00:00Z",
                    "end_time": "2026-01-01T00:00:05Z",
                    "device": "cpu",
                    "training_time": 5.0 + seed_index,
                    "n_params": 987,
                    "evaluation_set": "test",
                    "accuracy": f1 + 0.05,
                    "balanced_accuracy": f1 - 0.05,
                    "precision_macro": f1,
                    "recall_macro": f1,
                    "f1_macro": f1,
                    "loss": 1.0,
                    "val_loss": 0.5,
                    "val_accuracy": 0.75,
                    "warnings": "",
                    "failure_reason": "",
                }
            )
            config_rows.append(
                {
                    "config_id": config_id,
                    "space_key": _space_key(activation, optimizer, lr, init),
                    "validation_state": "VALID",
                    "validation_rationale": "",
                    "timestamp": timestamp,
                    "run_type": "prebenchmark",
                    "dataset": "wine_quality",
                    "dataset_version": "sha256-test",
                    "architecture": "wine_quality_mlp",
                    "arch_input_dim": 11,
                    "hidden_layers": "32|16",
                    "arch_output_dim": 7,
                    "activation": activation,
                    "optimizer": optimizer,
                    "lr": lr,
                    "init": init,
                    "loss": "cross_entropy",
                    "batch_size": 32,
                    "max_epochs": 10,
                    "seed": seed,
                    "device": "cpu",
                    "split_train": 0.6,
                    "split_val": 0.2,
                    "split_test": 0.2,
                    "split_stratify": True,
                    "scaler": "standard",
                    "class_weight": "N/A" if class_weight == "none" else class_weight,
                    "convergence_enabled": True,
                    "convergence_metric": "val_loss",
                    "convergence_patience": 20,
                    "convergence_min_delta": 1e-4,
                    "convergence_mode": "min",
                    "config_json": json.dumps(config, sort_keys=True),
                }
            )
            for epoch in range(1, 6):
                # Minimum validation loss at epoch 3; cumulative time 1.5 s.
                val_loss = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.7, 5: 0.9}[epoch]
                epoch_rows.append(
                    {
                        "run_id": run_id,
                        "config_id": config_id,
                        "epoch": epoch,
                        "train_loss": 1.1 - 0.1 * epoch,
                        "train_acc": 0.4 + 0.05 * epoch,
                        "val_loss": val_loss,
                        "val_acc": 0.5 + 0.02 * epoch,
                        "epoch_time": 0.5,
                    }
                )

    if include_failed:
        activation, optimizer, lr, init, class_weight = ("relu", "adam", 0.01, "he", "none")
        failed_seed = 999
        config = _config(activation, optimizer, lr, init, class_weight, failed_seed)
        config_id = "cfg-failed-run"
        master_rows.append(
            {
                "run_id": "20260101T000099Z-failed",
                "config_id": config_id,
                "space_key": _space_key(activation, optimizer, lr, init),
                "experiment_id": directory.name,
                "run_type": "prebenchmark",
                "dataset": "wine_quality",
                "dataset_version": "sha256-test",
                "architecture": "wine_quality_mlp",
                "activation": activation,
                "optimizer": optimizer,
                "lr": lr,
                "init": init,
                "loss_function": "cross_entropy",
                "batch_size": 32,
                "seed": failed_seed,
                "max_epochs": 10,
                "epochs_executed": 0,
                "execution_state": "FAILED",
                "training_time": "N/A",
                "evaluation_set": "test",
                "f1_macro": "N/A",
                "failure_reason": "non-finite loss",
            }
        )
        config_rows.append(
            {
                "config_id": config_id,
                "space_key": _space_key(activation, optimizer, lr, init),
                "validation_state": "VALID",
                "timestamp": timestamp,
                "activation": activation,
                "optimizer": optimizer,
                "lr": lr,
                "init": init,
                "class_weight": "N/A",
                "config_json": json.dumps(config, sort_keys=True),
            }
        )

    master_columns = [
        "run_id",
        "config_id",
        "space_key",
        "experiment_id",
        "run_type",
        "dataset",
        "dataset_version",
        "architecture",
        "activation",
        "optimizer",
        "lr",
        "init",
        "loss_function",
        "batch_size",
        "seed",
        "split_train",
        "split_val",
        "split_test",
        "split_stratify",
        "max_epochs",
        "epochs_executed",
        "execution_state",
        "start_time",
        "end_time",
        "device",
        "training_time",
        "n_params",
        "evaluation_set",
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
    ]
    config_columns = [
        "config_id",
        "space_key",
        "validation_state",
        "validation_rationale",
        "timestamp",
        "run_type",
        "dataset",
        "dataset_version",
        "architecture",
        "arch_input_dim",
        "hidden_layers",
        "arch_output_dim",
        "activation",
        "optimizer",
        "lr",
        "init",
        "loss",
        "batch_size",
        "max_epochs",
        "seed",
        "device",
        "split_train",
        "split_val",
        "split_test",
        "split_stratify",
        "scaler",
        "class_weight",
        "convergence_enabled",
        "convergence_metric",
        "convergence_patience",
        "convergence_min_delta",
        "convergence_mode",
        "config_json",
    ]
    epoch_columns = [
        "run_id",
        "config_id",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "epoch_time",
    ]

    _write_table(directory / f"master_{timestamp}.csv", master_rows, master_columns)
    _write_table(directory / f"configs_{timestamp}.csv", config_rows, config_columns)
    _write_table(directory / f"epochs_{timestamp}.csv", epoch_rows, epoch_columns)
    return directory


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    return write_export(tmp_path / "exp-synthetic")


def test_load_bundle_picks_latest_timestamp_and_joins_class_weight(export_dir: Path):
    write_export(export_dir, timestamp="20260102T000000Z", f1_shift=0.05)
    bundle = analysis.load_export_bundle(export_dir)

    assert bundle.timestamp == "20260102T000000Z"
    assert bundle.experiment_id == "exp-synthetic"
    assert len(bundle.runs) == 13
    assert bundle.runs["class_weight"].notna().all()
    assert set(bundle.runs["class_weight"]) == {"none", "balanced"}
    assert bundle.runs["group_key"].nunique() == 4
    assert bundle.runs["valid_run"].sum() == 12


def test_derive_convergence_metrics(export_dir: Path):
    bundle = analysis.load_export_bundle(export_dir)
    runs = analysis.derive_run_metrics(bundle)

    valid = runs[runs["valid_run"]]
    assert (valid["best_epoch"] == 3).all()
    assert (valid["best_val_loss"] == 0.5).all()
    assert (valid["time_to_best_epoch"] == 1.5).all()
    assert (valid["epochs_recorded"] == 5).all()
    assert (valid["epoch_time_mean"] == 0.5).all()

    failed = runs[~runs["valid_run"]]
    assert failed["best_epoch"].isna().all()


def test_summarize_groups_seeds_without_splitting_on_config_id(export_dir: Path):
    result = analysis.analyze_exports(export_dir)

    assert len(result.summary) == 4
    assert (result.summary["n_valid"] == 3).all()
    assert result.summary["n_runs"].sum() == 13
    assert result.summary["f1_macro_mean"].is_monotonic_decreasing
    best = result.summary.iloc[0]
    assert best["activation"] == "relu"
    assert best["class_weight"] == "none"
    assert best["f1_macro_mean"] == pytest.approx(0.81)
    assert best["f1_macro_std"] == pytest.approx(0.01)
    assert best["f1_macro_ci95_low"] < best["f1_macro_mean"] < best["f1_macro_ci95_high"]


def test_failed_runs_are_counted_but_excluded_from_metrics(export_dir: Path):
    result = analysis.analyze_exports(export_dir)
    best_group = result.summary.iloc[0]

    assert best_group["n_runs"] == 4
    assert best_group["n_valid"] == 3
    assert best_group["n_failed"] == 1
    assert any("no están en COMPLETED" in warning for warning in result.warnings)
    assert any("historial de épocas" in warning for warning in result.warnings)


def test_reference_and_pairwise_comparisons(export_dir: Path):
    result = analysis.analyze_exports(export_dir)
    reference = result.summary.iloc[0]

    assert result.reference_group_key == reference["group_key"]
    assert result.reference_label.startswith("relu | adam | lr=0.01")
    assert len(result.comparisons) == 3 * 4
    f1_comparisons = result.comparisons[result.comparisons["metric"] == "f1_macro"]
    assert len(f1_comparisons) == 3
    assert (f1_comparisons["diff"] < 0).all()
    assert (f1_comparisons["cliffs_delta"] == -1.0).all()
    assert (f1_comparisons["effect_size"] == "grande").all()
    assert (f1_comparisons["welch_p"] < 0.05).all()


def test_reference_accepts_a_member_config_id(export_dir: Path):
    bundle = analysis.load_export_bundle(export_dir)
    member = bundle.runs[bundle.runs["class_weight"] == "balanced"].iloc[0]

    result = analysis.analyze_exports(export_dir, reference_config_id=str(member["config_id"]))

    assert result.reference_group_key == str(member["group_key"])
    assert "referencia explícita" in result.reference_reason


def test_reference_rejects_unknown_id(export_dir: Path):
    with pytest.raises(ValueError, match="no es un group_key ni un config_id"):
        analysis.analyze_exports(export_dir, reference_config_id="cfg-does-not-exist")


def test_marginal_effects_group_by_dimension(export_dir: Path):
    result = analysis.analyze_exports(export_dir)
    marginals = result.marginals

    assert set(marginals["dimension"]) == {"activation", "optimizer", "lr", "init", "class_weight"}
    activation = marginals[marginals["dimension"] == "activation"]
    assert set(activation["level"]) == {"relu", "tanh"}
    assert (activation["n_configs"] == 2).all()
    assert (activation["n_runs"] == 6).all()
    assert activation["kruskal_p"].notna().all()

    class_weight = marginals[marginals["dimension"] == "class_weight"]
    assert set(class_weight["level"]) == {"none", "balanced"}
    assert (class_weight["n_configs"] == 2).all()


def test_missing_required_table_raises(tmp_path: Path):
    empty = tmp_path / "exp-empty"
    empty.mkdir()
    (empty / "master_20260101T000000Z.csv").write_text("run_id\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="faltan tablas requeridas"):
        analysis.load_export_bundle(empty)


def test_discover_export_files_ignores_unknown_tables(export_dir: Path):
    (export_dir / "notes_20260101T000000Z.csv").write_text("x\n", encoding="utf-8")
    files = analysis.discover_export_files(export_dir)

    assert set(files) == {"master", "configs", "epochs"}
