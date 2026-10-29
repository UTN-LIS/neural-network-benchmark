"""Unit tests for the configuration model (AC1, AC12)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from prebenchmark.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONVERGENCE_METRIC,
    DEFAULT_CONVERGENCE_MIN_DELTA,
    DEFAULT_CONVERGENCE_MODE,
    DEFAULT_CONVERGENCE_PATIENCE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_SEED,
    KNOWN_ACTIVATIONS,
    KNOWN_INITIALIZATIONS,
    KNOWN_OPTIMIZERS,
    ArchitectureSpec,
    ConvergenceSpec,
    DatasetSpec,
    ExperimentConfig,
    PreprocessingSpec,
    SplitSpec,
    resolve_convergence,
)


def test_default_protocol_values() -> None:
    cfg = ExperimentConfig()
    assert cfg.batch_size == DEFAULT_BATCH_SIZE == 32
    assert cfg.max_epochs == DEFAULT_MAX_EPOCHS == 500
    assert cfg.seed == DEFAULT_SEED == 42
    assert cfg.device == DEFAULT_DEVICE == "auto"
    assert cfg.run_type == "prebenchmark"
    assert cfg.loss == "cross_entropy"


def test_default_split_proportions_are_60_20_20() -> None:
    split = ExperimentConfig().split
    assert split.proportions() == pytest.approx((0.6, 0.2, 0.2))
    assert split.stratify is True


def test_default_convergence_is_off() -> None:
    convergence = ExperimentConfig().convergence
    assert convergence.enabled is False
    assert convergence.metric is None
    assert convergence.patience is None
    assert convergence.min_delta is None


def test_resolve_convergence_fills_only_unspecified_fields() -> None:
    partial = ConvergenceSpec(enabled=True, metric="train_loss")
    resolved = resolve_convergence(partial)
    assert resolved.enabled is True
    assert resolved.metric == "train_loss"  # explicitly declared, untouched
    assert resolved.patience == DEFAULT_CONVERGENCE_PATIENCE
    assert resolved.min_delta == DEFAULT_CONVERGENCE_MIN_DELTA
    assert resolved.mode == DEFAULT_CONVERGENCE_MODE


def test_resolve_convergence_uses_documented_default_metric() -> None:
    resolved = resolve_convergence(ConvergenceSpec(enabled=True))
    assert resolved.metric == DEFAULT_CONVERGENCE_METRIC


def test_class_weight_is_recorded_and_defaults_to_none() -> None:
    cfg = ExperimentConfig()
    assert cfg.preprocessing.class_weight is None
    balanced = replace(
        cfg, preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced")
    )
    assert balanced.preprocessing.class_weight == "balanced"
    assert balanced.to_dict()["preprocessing"]["class_weight"] == "balanced"


def test_default_config_matches_baseline_reference() -> None:
    cfg = ExperimentConfig()
    assert cfg.activation == "relu"
    assert cfg.optimizer == "adam"
    assert cfg.lr == 0.001
    assert cfg.init == "he"


def test_to_dict_is_complete_and_plain() -> None:
    data = ExperimentConfig().to_dict()
    assert set(data) == {
        "run_type",
        "dataset",
        "architecture",
        "activation",
        "optimizer",
        "lr",
        "init",
        "loss",
        "batch_size",
        "max_epochs",
        "seed",
        "split",
        "preprocessing",
        "convergence",
        "device",
    }
    # Nested specs must serialize as plain dicts, not dataclasses.
    for key in ("dataset", "architecture", "split", "preprocessing", "convergence"):
        assert isinstance(data[key], dict)


def test_config_round_trip_preserves_all_values() -> None:
    original = ExperimentConfig(
        run_type="benchmark",
        dataset=DatasetSpec(name="wine_quality", version="uci-v1", n_features=11),
        architecture=ArchitectureSpec(hidden_layers=[64, 32, 16], output_dim=7),
        activation="tanh",
        optimizer="sgd",
        lr=0.01,
        init="xavier",
        loss="cross_entropy",
        batch_size=64,
        max_epochs=250,
        seed=7,
        split=SplitSpec(train=0.7, val=0.15, test=0.15, stratify=False),
        preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced"),
        convergence=ConvergenceSpec(
            enabled=True, metric="val_loss", patience=10, min_delta=1e-4, mode="min"
        ),
        device="cpu",
    )
    restored = ExperimentConfig.from_dict(original.to_dict())
    assert restored == original
    assert restored.to_dict() == original.to_dict()


def test_from_dict_does_not_alias_hidden_layers() -> None:
    data = ExperimentConfig().to_dict()
    data["architecture"]["hidden_layers"] = [128, 64]
    cfg = ExperimentConfig.from_dict(data)
    data["architecture"]["hidden_layers"].append(999)
    assert cfg.architecture.hidden_layers == [128, 64]


@pytest.mark.parametrize(
    "spec",
    [
        DatasetSpec(),
        ArchitectureSpec(),
        SplitSpec(),
        PreprocessingSpec(),
        ConvergenceSpec(),
    ],
)
def test_spec_round_trip(spec) -> None:
    assert type(spec).from_dict(spec.to_dict()) == spec


def test_name_registries_hold_documented_initial_sets() -> None:
    assert set(KNOWN_ACTIVATIONS) == {"relu", "sigmoid", "tanh"}
    assert set(KNOWN_OPTIMIZERS) == {"sgd", "adam"}
    assert set(KNOWN_INITIALIZATIONS) == {"xavier", "he"}
