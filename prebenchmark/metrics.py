"""Cálculo de métricas para el entrenamiento y la evaluación final."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_final_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float | None = None,
) -> Dict[str, Any]:
    """Calcula el conjunto final de métricas con promedio macro."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def evaluate(
    model: nn.Module,
    data_loader: Any,
    device: str = "cpu",
    criterion: nn.Module | None = None,
) -> Dict[str, Any]:
    """Evalúa el modelo sobre el cargador de datos.

    Se detiene en el primer lote cuya pérdida no es finita, registrando
    ``non_finite=True`` y ``non_finite_source`` para que el invocador pueda marcar
    la ejecución en lugar de propagar una métrica corrupta.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    non_finite = False
    non_finite_source: str | None = None
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not bool(torch.isfinite(loss).item()):
                non_finite = True
                non_finite_source = "loss"
                break
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    loss = total_loss / len(all_labels) if all_labels else 0.0
    if all_labels:
        metrics = compute_final_metrics(np.array(all_labels), np.array(all_preds), loss)
    else:
        # No finito detectado antes de recolectar predicción alguna: se reportan
        # los indicadores sin pedir a sklearn que evalúe arreglos vacíos.
        metrics = {"loss": loss, "accuracy": 0.0}
    metrics["non_finite"] = non_finite
    metrics["non_finite_source"] = non_finite_source
    return metrics


def epoch_train_step(
    model: nn.Module,
    data_loader: Any,
    optimizer: Any,
    criterion: nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Ejecuta una época de entrenamiento, deteniéndose en la primera pérdida no finita.

    El paso del optimizador se omite ante una pérdida no finita para que un gradiente
    corrupto nunca contamine los pesos. El mapeo devuelto siempre contiene
    ``train_loss``/``train_acc`` además de ``non_finite`` (bool), ``non_finite_batch``
    (índice dentro de la época) y ``non_finite_source``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    non_finite = False
    non_finite_batch: int | None = None
    non_finite_source: str | None = None
    for batch_index, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if not bool(torch.isfinite(loss).item()):
            non_finite = True
            non_finite_batch = batch_index
            non_finite_source = "train_loss"
            break
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return {
        "train_loss": total_loss / total if total else float("nan"),
        "train_acc": correct / total if total else float("nan"),
        "non_finite": non_finite,
        "non_finite_batch": non_finite_batch,
        "non_finite_source": non_finite_source,
    }
