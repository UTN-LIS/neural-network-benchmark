"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from prebenchmark import metrics
from torch.utils.data import DataLoader, TensorDataset


def test_compute_final_metrics_macro():
    y_true = np.array([0, 1, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 0, 2, 1])
    res = metrics.compute_final_metrics(y_true, y_pred)
    assert "accuracy" in res
    assert "balanced_accuracy" in res
    assert "precision_macro" in res
    assert "recall_macro" in res
    assert "f1_macro" in res
    assert "confusion_matrix" in res


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


def _loader_with_extreme_input() -> DataLoader:
    inputs = torch.zeros(4, 3)
    inputs[0, 0] = float("inf")
    labels = torch.tensor([0, 1, 2, 0])
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


def test_epoch_train_step_flags_non_finite_loss():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    result = metrics.epoch_train_step(
        model, _loader_with_extreme_input(), optimizer, criterion, device="cpu"
    )
    assert result["non_finite"] is True
    assert result["non_finite_source"] == "train_loss"
    assert result["non_finite_batch"] == 0


def test_evaluate_flags_non_finite_loss():
    model = _TinyModel()
    criterion = nn.CrossEntropyLoss()
    result = metrics.evaluate(
        model, _loader_with_extreme_input(), device="cpu", criterion=criterion
    )
    assert result["non_finite"] is True
    assert result["non_finite_source"] == "loss"


def test_epoch_train_step_finite_path_reports_no_flags():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),
        batch_size=4,
    )
    result = metrics.epoch_train_step(model, loader, optimizer, criterion, device="cpu")
    assert result["non_finite"] is False
    assert result["non_finite_batch"] is None
    assert result["non_finite_source"] is None
