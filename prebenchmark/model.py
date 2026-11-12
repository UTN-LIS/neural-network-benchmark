"""Construcción del modelo e inicialización de pesos.

Construye la arquitectura MLP parametrizada por la especificación de arquitectura
y la activación. Implementa la inicialización He/Kaiming y Xavier/Glorot según lo
especificado.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .config import ArchitectureSpec, ExperimentConfig


def build_model(
    arch: ArchitectureSpec | None = None,
    activation: str | None = None,
    *,
    cfg: ExperimentConfig | None = None,
) -> nn.Module:
    """Construye el MLP a partir de la arquitectura y la activación."""
    if cfg is not None:
        arch = cfg.architecture if arch is None else arch
        activation = cfg.activation if activation is None else activation

    if arch is None:
        arch = ArchitectureSpec()

    act_name = (activation or "relu").lower()

    layers: list[nn.Module] = []
    # Construye las capas ocultas
    in_dim = arch.input_dim
    for h in arch.hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        if act_name == "relu":
            layers.append(nn.ReLU())
        elif act_name == "sigmoid":
            layers.append(nn.Sigmoid())
        elif act_name == "tanh":
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, arch.output_dim))

    return nn.Sequential(*layers)


def initialize_weights(
    model: nn.Module,
    init: str,
    activation: str,
) -> nn.Module:
    """Inicializa los pesos (Xavier/Glorot o He/Kaiming). Los sesgos se establecen en cero."""
    init_name = (init or "xavier").lower()
    act_name = (activation or "relu").lower()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_name == "he" or init_name == "kaiming":
                nonlinearity: Literal["relu", "tanh", "sigmoid"] = "relu"
                if act_name == "tanh":
                    nonlinearity = "tanh"
                elif act_name == "sigmoid":
                    nonlinearity = "sigmoid"
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def parameters_are_finite(model: nn.Module) -> bool:
    """Verdadero cuando todos los parámetros entrenables son finitos (sin pesos NaN/Inf)."""
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())
