"""Unit tests for stable/unique identifiers (AC3, AC12)."""

from __future__ import annotations

import re
from dataclasses import replace

import prebenchmark.identity as identity_module
import prebenchmark.space as space_module
import pytest
from prebenchmark.config import ExperimentConfig, PreprocessingSpec
from prebenchmark.identity import (
    compute_config_id,
    compute_space_key,
    new_decision_id,
    new_experiment_id,
    new_run_id,
    utc_timestamp,
)

# ---------------------------------------------------------------------------
# config_id: stable, content-addressed, sensitive to every field
# ---------------------------------------------------------------------------


def test_config_id_is_stable_for_identical_configs() -> None:
    assert compute_config_id(ExperimentConfig()) == compute_config_id(ExperimentConfig())


def test_config_id_survives_serialization_round_trip() -> None:
    original = ExperimentConfig(activation="tanh", optimizer="sgd", lr=0.01, init="xavier")
    restored = ExperimentConfig.from_dict(original.to_dict())
    assert compute_config_id(original) == compute_config_id(restored)


def test_config_id_has_expected_prefix_and_length() -> None:
    config_id = compute_config_id(ExperimentConfig())
    assert config_id.startswith("cfg-v1-")
    assert re.fullmatch(r"cfg-v1-[0-9a-f]{12}", config_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_type", "benchmark"),
        ("activation", "sigmoid"),
        ("optimizer", "sgd"),
        ("lr", 0.1),
        ("init", "xavier"),
        ("loss", "mse"),
        ("batch_size", 64),
        ("max_epochs", 250),
        ("seed", 7),
        ("device", "cpu"),
    ],
)
def test_config_id_changes_with_any_parameter(field: str, value: object) -> None:
    base = ExperimentConfig()
    assert compute_config_id(base) != compute_config_id(replace(base, **{field: value}))


def test_config_id_includes_class_weight_unlike_space_key() -> None:
    base = ExperimentConfig()
    balanced = replace(
        base, preprocessing=PreprocessingSpec(scaler="standard", class_weight="balanced")
    )
    # Full-config identity distinguishes them...
    assert compute_config_id(base) != compute_config_id(balanced)
    # ...but space membership deliberately does not.
    assert compute_space_key(base) == compute_space_key(balanced)


# ---------------------------------------------------------------------------
# compute_space_key is imported from space (single definition)
# ---------------------------------------------------------------------------


def test_compute_space_key_is_not_redefined() -> None:
    assert compute_space_key is space_module.compute_space_key
    assert identity_module.compute_space_key is space_module.compute_space_key


# ---------------------------------------------------------------------------
# run / experiment / decision ids: unique and well-formed
# ---------------------------------------------------------------------------


def test_run_ids_are_unique() -> None:
    ids = {new_run_id() for _ in range(300)}
    assert len(ids) == 300


def test_run_id_format() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", new_run_id())


def test_run_id_uses_supplied_timestamp_but_still_unique() -> None:
    stamp = utc_timestamp()
    assert new_run_id(stamp).startswith(stamp + "-")
    assert new_run_id(stamp) != new_run_id(stamp)


def test_experiment_id_is_unique_and_can_be_labelled() -> None:
    assert len({new_experiment_id() for _ in range(50)}) == 50
    labelled = new_experiment_id("Wine Quality Run")
    assert labelled.startswith("exp-wine-quality-run-")
    assert re.fullmatch(r"exp-wine-quality-run-\d{8}T\d{6}Z-[0-9a-f]{6}", labelled)


def test_decision_id_format_and_uniqueness() -> None:
    assert len({new_decision_id() for _ in range(50)}) == 50
    assert re.fullmatch(r"dec-\d{8}T\d{6}Z-[0-9a-f]{6}", new_decision_id())


def test_utc_timestamp_is_lexicographically_sortable() -> None:
    earlier = utc_timestamp()
    later = utc_timestamp()
    assert earlier <= later
    assert re.fullmatch(r"\d{8}T\d{6}Z", earlier)
