"""Modelos de entrada de la API de la UI.

La validación de dominio NO se duplica aquí: los valores permitidos y las reglas
metodológicas viven en ``prebenchmark.space`` y se aplican con
``space.validate_config``. Pydantic solo cubre la forma del request.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ClassWeight = Literal["none", "balanced"]
Device = Literal["auto", "cpu", "cuda"]


class RunRequest(BaseModel):
    dataset: str = Field(min_length=1)
    activation: str = Field(min_length=1)
    optimizer: str = Field(min_length=1)
    lr: float = Field(gt=0)
    init: str = Field(min_length=1)
    class_weight: ClassWeight = "none"
    seed: int = 42
    max_epochs: int = Field(default=500, ge=1, le=5000)
    # La UI enciende el early stopping por defecto: las suites completas se
    # corrieron con el criterio habilitado y sin él una corrida puede consumir
    # el tope de 500 épocas (hasta ~122 s medidos en satimage).
    early_stop: bool = True
    # El set de prueba queda reservado. Solo se devuelve si el cliente lo pide
    # de forma explícita, y la respuesta lo marca como tal.
    include_test: bool = False
    device: Device = "auto"


class ValidationOut(BaseModel):
    state: str
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class JobCreated(BaseModel):
    job_id: str
    status: str
    validation: ValidationOut


class JobStatus(BaseModel):
    job_id: str
    status: str
    dataset: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    epochs: int = 0
    execution_state: str | None = None
    cancel_requested: bool = False
    error: str | None = None
    result: dict[str, Any] | None = None
