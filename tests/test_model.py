"""Tests for model module."""

from __future__ import annotations

import torch

from prebenchmark import config, model


def test_build_model_architecture():
    m = model.build_model(config.ArchitectureSpec(input_dim=11, hidden_layers=[32, 16], output_dim=7), activation="relu")
    seq = m
    # check structure
    assert isinstance(seq, torch.nn.Sequential)


def test_initialize_weights_he():
    m = model.build_model()
    model.initialize_weights(m, "he", "relu")
    # just ensure no crash
    assert m is not None


def test_initialize_weights_xavier():
    m = model.build_model(activation="tanh")
    model.initialize_weights(m, "xavier", "tanh")
    assert m is not None


def test_parameters_are_finite_true_after_initialization():
    m = model.build_model()
    model.initialize_weights(m, "he", "relu")
    assert model.parameters_are_finite(m) is True


def test_parameters_are_finite_false_when_weight_is_nan():
    m = model.build_model()
    with torch.no_grad():
        next(m.parameters())[0, 0] = float("nan")
    assert model.parameters_are_finite(m) is False
