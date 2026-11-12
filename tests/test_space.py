"""Unit tests for space generation and configuration validation (AC1, AC2, AC12)."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import prebenchmark.space as space_module
import pytest
from prebenchmark.config import (
    ArchitectureSpec,
    ConvergenceSpec,
    ExperimentConfig,
    PreprocessingSpec,
    SplitSpec,
)
from prebenchmark.space import (
    INVALID,
    SPACE_DIMENSIONS,
    VALID,
    WARNING,
    ParameterSpace,
    compute_space_key,
    default_space,
    generate_configs,
    space_key_dims,
    validate_config,
)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generates_exactly_36_combinations() -> None:
    configs = generate_configs(default_space())
    assert len(configs) == 36


def test_generated_combinations_are_the_full_cartesian_product() -> None:
    active = default_space()
    configs = generate_configs(active)
    observed = {(c.activation, c.optimizer, c.lr, c.init) for c in configs}
    expected = set(
        product(
            active.activations,
            active.optimizers,
            active.learning_rates,
            active.initializations,
        )
    )
    assert observed == expected  # no silent drops, no injected combinations


def test_space_keys_are_unique_for_all_36() -> None:
    configs = generate_configs(default_space())
    keys = {compute_space_key(c) for c in configs}
    assert len(keys) == 36


def test_generated_configs_are_independent_copies() -> None:
    configs = generate_configs(default_space())
    configs[0].split.train = 0.9
    configs[0].architecture.hidden_layers.append(999)
    assert configs[1].split.train == 0.6
    assert configs[1].architecture.hidden_layers == [32, 16]


def test_extensibility_scales_without_core_logic_changes(monkeypatch) -> None:
    extended = ParameterSpace(activations=["relu", "sigmoid", "tanh", "gelu"])
    configs = generate_configs(extended)
    assert len(configs) == 48
    assert len({compute_space_key(c) for c in configs}) == 48

    # A new name is unknown until it is registered (data, not logic).
    gelu = [c for c in configs if c.activation == "gelu"]
    assert all(validate_config(c).state == INVALID for c in gelu)

    monkeypatch.setattr(
        space_module,
        "KNOWN_ACTIVATIONS",
        ("relu", "sigmoid", "tanh", "gelu"),
    )
    assert all(validate_config(c).is_executable() for c in gelu)


# ---------------------------------------------------------------------------
# Space key: class_weight is excluded, the 4 dimensions are included
# ---------------------------------------------------------------------------


def test_space_key_excludes_class_weight() -> None:
    cfg = ExperimentConfig()
    balanced = replace(
        cfg, preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced")
    )
    assert space_key_dims(cfg) == space_key_dims(balanced)
    assert compute_space_key(cfg) == compute_space_key(balanced)


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation", "sigmoid"),
        ("optimizer", "sgd"),
        ("lr", 0.1),
        ("init", "xavier"),
    ],
)
def test_space_key_is_sensitive_to_each_space_dimension(field, value) -> None:
    base = ExperimentConfig()
    changed = replace(base, **{field: value})
    assert compute_space_key(base) != compute_space_key(changed)


def test_space_dimensions_constant_matches_the_four_free_parameters() -> None:
    assert SPACE_DIMENSIONS == ("activation", "optimizer", "lr", "init")


# ---------------------------------------------------------------------------
# Validation states
# ---------------------------------------------------------------------------


def test_none_of_the_36_combinations_is_invalid() -> None:
    results = [validate_config(c) for c in generate_configs(default_space())]
    assert all(r.state != INVALID for r in results)


def test_default_config_is_valid() -> None:
    result = validate_config(ExperimentConfig())
    assert result.state == VALID
    assert result.rationale == []
    assert result.is_executable() is True


def test_high_learning_rate_is_warning_not_invalid() -> None:
    adm = replace(ExperimentConfig(), optimizer="adam", lr=0.1)
    sgd = replace(ExperimentConfig(), optimizer="sgd", lr=0.1)
    for cfg in (adm, sgd):
        result = validate_config(cfg)
        assert result.state == WARNING
        assert result.rationale  # non-empty explanation
        assert result.is_executable() is True


def test_non_recommended_pairings_are_warnings() -> None:
    he_sigmoid = replace(ExperimentConfig(), activation="sigmoid", init="he")
    xavier_relu = replace(ExperimentConfig(), activation="relu", init="xavier")
    for cfg in (he_sigmoid, xavier_relu):
        result = validate_config(cfg)
        assert result.state == WARNING
        assert result.is_executable() is True


def test_adam_with_high_lr_is_not_rejected_by_convention() -> None:
    cfg = replace(ExperimentConfig(), optimizer="adam", lr=0.1)
    result = validate_config(cfg)
    assert result.state != INVALID
    assert result.is_executable() is True


def test_enabled_convergence_with_unspecified_fields_is_executable() -> None:
    cfg = replace(
        ExperimentConfig(),
        convergence=ConvergenceSpec(enabled=True, metric=None),
    )
    result = validate_config(cfg)
    assert result.state != INVALID
    assert result.is_executable() is True


@pytest.mark.parametrize(
    "cfg,expected_rule",
    [
        (replace(ExperimentConfig(), activation="gelu"), "unknown_activation"),
        (replace(ExperimentConfig(), optimizer="rmsprop"), "unknown_optimizer"),
        (replace(ExperimentConfig(), init="orthogonal"), "unknown_init"),
        (replace(ExperimentConfig(), loss="mse"), "unknown_loss"),
        (replace(ExperimentConfig(), batch_size=0), "batch_size_below_one"),
        (replace(ExperimentConfig(), max_epochs=0), "max_epochs_below_one"),
        (replace(ExperimentConfig(), lr=0.0), "lr_not_positive"),
        (
            replace(ExperimentConfig(), split=SplitSpec(train=0.5, val=0.2, test=0.2)),
            "split_not_summing_to_one",
        ),
        (
            replace(
                ExperimentConfig(),
                architecture=ArchitectureSpec(input_dim=5, output_dim=7),
            ),
            "architecture_input_mismatch",
        ),
        (
            replace(
                ExperimentConfig(),
                convergence=ConvergenceSpec(enabled=True, metric="accuracy"),
            ),
            "convergence_incomplete",
        ),
    ],
)
def test_invalid_configurations_are_blocked_with_reason(cfg, expected_rule) -> None:
    result = validate_config(cfg)
    assert result.state == INVALID
    assert result.is_executable() is False
    assert result.rationale  # explicit reason recorded
    assert expected_rule in result.fired_rules


def test_custom_rules_are_configurable() -> None:
    # Passing an explicit empty rule list must not apply project defaults.
    result = validate_config(replace(ExperimentConfig(), lr=-1.0), rules=[])
    assert result.state == VALID


def test_validation_result_to_dict() -> None:
    result = validate_config(replace(ExperimentConfig(), optimizer="adam", lr=0.1))
    payload = result.to_dict()
    assert payload["state"] == WARNING
    assert payload["warnings"] == result.warnings
    assert payload["fired_rules"] == result.fired_rules
