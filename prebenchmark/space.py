"""Espacio de parámetros explícito, generación de 36 combinaciones y prevalidación.

El conjunto de candidatos se produce a partir de un :class:`ParameterSpace` explícito
mediante ``itertools.product`` -- ningún experimento individual está codificado de forma fija.
La validación clasifica cada configuración como ``VALID``, ``WARNING`` o ``INVALID`` con una
justificación explícita. Las reglas por defecto dejan toda combinación generada como ejecutable
(como máximo WARNING); nada se rechaza solo por convención.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Callable, Iterable, Sequence

from .config import (
    BASELINE_ACTIVATION,
    BASELINE_INIT,
    BASELINE_LR,
    BASELINE_OPTIMIZER,
    DEFAULT_DATASET,
    KNOWN_ACTIVATIONS,
    KNOWN_CONVERGENCE_METRICS,
    KNOWN_CONVERGENCE_MODES,
    KNOWN_DEVICES,
    KNOWN_INITIALIZATIONS,
    KNOWN_LOSSES,
    KNOWN_OPTIMIZERS,
    KNOWN_RUN_TYPES,
    KNOWN_SCALERS,
    ExperimentConfig,
    PreprocessingSpec,
)

# ---------------------------------------------------------------------------
# Estados y severidades de validación
# ---------------------------------------------------------------------------

VALID = "VALID"
WARNING = "WARNING"
INVALID = "INVALID"
_SEVERITIES = (WARNING, INVALID)

# Umbral de tasa de aprendizaje "alta" usado por las reglas WARNING por defecto. Se mantiene como
# una constante con nombre para que el valor por defecto documentado sea visible y configurable.
HIGH_LR_THRESHOLD = 0.1
_SPLIT_TOLERANCE = 1e-9

# Las cuatro dimensiones libres que definen la pertenencia al espacio. class_weight y todos
# los ajustes de protocolo se excluyen deliberadamente.
SPACE_DIMENSIONS: tuple[str, ...] = ("activation", "optimizer", "lr", "init")


# ---------------------------------------------------------------------------
# Modelo de reglas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationRule:
    """Un predicado con nombre sobre una configuración.

    ``severity`` es ``WARNING`` (ejecutable, riesgo registrado) o
    ``INVALID`` (ejecución bloqueada). ``predicate`` devuelve True cuando la regla
    se activa.
    """

    code: str
    severity: str
    message: str
    predicate: Callable[[ExperimentConfig], bool]


@dataclass
class ValidationResult:
    """Resultado de validar una configuración."""

    state: str
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fired_rules: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.state == VALID

    @property
    def is_warning(self) -> bool:
        return self.state == WARNING

    @property
    def is_invalid(self) -> bool:
        return self.state == INVALID

    def is_executable(self) -> bool:
        """Las configuraciones WARNING y VALID pueden ejecutarse; las INVALID no."""
        return self.state in (VALID, WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "fired_rules": list(self.fired_rules),
        }


# ---------------------------------------------------------------------------
# Reglas por defecto -- INVALID (solo restricciones técnicas/metodológicas)
# ---------------------------------------------------------------------------


def _invalid_unknown_activation(cfg: ExperimentConfig) -> bool:
    return cfg.activation not in KNOWN_ACTIVATIONS


def _invalid_unknown_optimizer(cfg: ExperimentConfig) -> bool:
    return cfg.optimizer not in KNOWN_OPTIMIZERS


def _invalid_unknown_init(cfg: ExperimentConfig) -> bool:
    return cfg.init not in KNOWN_INITIALIZATIONS


def _invalid_unknown_loss(cfg: ExperimentConfig) -> bool:
    return cfg.loss not in KNOWN_LOSSES


def _invalid_unknown_scaler(cfg: ExperimentConfig) -> bool:
    return cfg.preprocessing.scaler not in KNOWN_SCALERS


def _invalid_unknown_run_type(cfg: ExperimentConfig) -> bool:
    return cfg.run_type not in KNOWN_RUN_TYPES


def _invalid_unknown_device(cfg: ExperimentConfig) -> bool:
    return cfg.device not in KNOWN_DEVICES


def _invalid_arch_input_mismatch(cfg: ExperimentConfig) -> bool:
    return cfg.architecture.input_dim != cfg.dataset.n_features


def _invalid_arch_output_too_small(cfg: ExperimentConfig) -> bool:
    return cfg.architecture.output_dim < 2


def _invalid_split_not_summing_to_one(cfg: ExperimentConfig) -> bool:
    total = sum(cfg.split.proportions())
    return abs(total - 1.0) > _SPLIT_TOLERANCE


def _invalid_split_negative(cfg: ExperimentConfig) -> bool:
    return any(p < 0 for p in cfg.split.proportions())


def _invalid_batch_size_below_one(cfg: ExperimentConfig) -> bool:
    return cfg.batch_size < 1


def _invalid_max_epochs_below_one(cfg: ExperimentConfig) -> bool:
    return cfg.max_epochs < 1


def _invalid_lr_not_positive(cfg: ExperimentConfig) -> bool:
    return cfg.lr <= 0


def _invalid_convergence_incomplete(cfg: ExperimentConfig) -> bool:
    # Un criterio habilitado puede dejar campos sin especificar (los valores por defecto documentados
    # los completan), pero todo valor declarado explícitamente debe ser reconocido. No se usan
    # umbrales ocultos: los campos sin especificar se resuelven con los valores por defecto registrados.
    if not cfg.convergence.enabled:
        return False
    metric = cfg.convergence.metric
    if metric is not None and metric not in KNOWN_CONVERGENCE_METRICS:
        return True
    return cfg.convergence.mode not in KNOWN_CONVERGENCE_MODES


# ---------------------------------------------------------------------------
# Reglas por defecto -- WARNING (ejecutable, riesgo registrado)
# ---------------------------------------------------------------------------


def _warn_high_lr_adam(cfg: ExperimentConfig) -> bool:
    return cfg.optimizer == "adam" and cfg.lr >= HIGH_LR_THRESHOLD


def _warn_high_lr_sgd(cfg: ExperimentConfig) -> bool:
    return cfg.optimizer == "sgd" and cfg.lr >= HIGH_LR_THRESHOLD


def _warn_he_with_non_relu(cfg: ExperimentConfig) -> bool:
    return cfg.init == "he" and cfg.activation in ("sigmoid", "tanh")


def _warn_xavier_with_relu(cfg: ExperimentConfig) -> bool:
    return cfg.init == "xavier" and cfg.activation == "relu"


DEFAULT_VALIDATION_RULES: tuple[ValidationRule, ...] = (
    ValidationRule(
        "unknown_activation", INVALID,
        "activation name is not registered",
        _invalid_unknown_activation,
    ),
    ValidationRule(
        "unknown_optimizer", INVALID,
        "optimizer name is not registered",
        _invalid_unknown_optimizer,
    ),
    ValidationRule(
        "unknown_init", INVALID,
        "weight initialization name is not registered",
        _invalid_unknown_init,
    ),
    ValidationRule(
        "unknown_loss", INVALID,
        "loss function name is not registered",
        _invalid_unknown_loss,
    ),
    ValidationRule(
        "unknown_scaler", INVALID,
        "preprocessing scaler is not registered",
        _invalid_unknown_scaler,
    ),
    ValidationRule(
        "unknown_run_type", INVALID,
        "run_type must be 'prebenchmark' or 'benchmark'",
        _invalid_unknown_run_type,
    ),
    ValidationRule(
        "unknown_device", INVALID,
        "device must be one of 'auto', 'cpu', 'cuda'",
        _invalid_unknown_device,
    ),
    ValidationRule(
        "architecture_input_mismatch", INVALID,
        "architecture input_dim does not match dataset n_features",
        _invalid_arch_input_mismatch,
    ),
    ValidationRule(
        "architecture_output_too_small", INVALID,
        "architecture output_dim must be at least 2 classes",
        _invalid_arch_output_too_small,
    ),
    ValidationRule(
        "split_not_summing_to_one", INVALID,
        "train/val/test proportions must sum to 1.0",
        _invalid_split_not_summing_to_one,
    ),
    ValidationRule(
        "split_negative", INVALID,
        "split proportions must be non-negative",
        _invalid_split_negative,
    ),
    ValidationRule(
        "batch_size_below_one", INVALID,
        "batch_size must be >= 1",
        _invalid_batch_size_below_one,
    ),
    ValidationRule(
        "max_epochs_below_one", INVALID,
        "max_epochs must be >= 1",
        _invalid_max_epochs_below_one,
    ),
    ValidationRule(
        "lr_not_positive", INVALID,
        "learning rate must be > 0",
        _invalid_lr_not_positive,
    ),
    ValidationRule(
        "convergence_incomplete", INVALID,
        "convergence is enabled but names an unrecognized metric or mode",
        _invalid_convergence_incomplete,
    ),
    ValidationRule(
        "high_lr_adam", WARNING,
        "Adam with learning rate >= 0.1 may be unstable",
        _warn_high_lr_adam,
    ),
    ValidationRule(
        "high_lr_sgd", WARNING,
        "SGD with learning rate >= 0.1 may diverge",
        _warn_high_lr_sgd,
    ),
    ValidationRule(
        "he_with_non_relu", WARNING,
        "He/Kaiming initialization is not conventionally recommended for "
        "sigmoid/tanh activations",
        _warn_he_with_non_relu,
    ),
    ValidationRule(
        "xavier_with_relu", WARNING,
        "Xavier/Glorot initialization is not conventionally recommended for "
        "ReLU activations",
        _warn_xavier_with_relu,
    ),
)


def validate_config(
    cfg: ExperimentConfig,
    rules: Sequence[ValidationRule] | Iterable[ValidationRule] | None = None,
) -> ValidationResult:
    """Clasifica ``cfg`` en exactamente uno de VALID / WARNING / INVALID.

    Cualquier regla INVALID activada domina; en caso contrario, cualquier WARNING produce WARNING;
    de lo contrario, VALID. Todos los mensajes activados se registran como justificación.
    """
    active_rules = list(DEFAULT_VALIDATION_RULES if rules is None else rules)
    fired = [rule for rule in active_rules if rule.predicate(cfg)]
    errors = [rule.message for rule in fired if rule.severity == INVALID]
    warnings = [rule.message for rule in fired if rule.severity == WARNING]

    if errors:
        state = INVALID
        rationale = errors + warnings
    elif warnings:
        state = WARNING
        rationale = warnings
    else:
        state = VALID
        rationale = []

    return ValidationResult(
        state=state,
        rationale=rationale,
        warnings=warnings,
        errors=errors,
        fired_rules=[rule.code for rule in fired],
    )


# ---------------------------------------------------------------------------
# Espacio de parámetros
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """Espacio de candidatos explícito sobre las cuatro dimensiones libres.

    El espacio de protocolo inicial produce 3 x 2 x 3 x 2 = 36 combinaciones.
    Agregar valores (por ejemplo una cuarta activación) extiende el espacio sin
    tocar la lógica de validación ni de generación.
    """

    activations: list[str] = field(
        default_factory=lambda: ["relu", "sigmoid", "tanh"]
    )
    optimizers: list[str] = field(default_factory=lambda: ["sgd", "adam"])
    learning_rates: list[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1]
    )
    initializations: list[str] = field(default_factory=lambda: ["xavier", "he"])
    base: ExperimentConfig = field(default_factory=ExperimentConfig)

    def size(self) -> int:
        return (
            len(self.activations)
            * len(self.optimizers)
            * len(self.learning_rates)
            * len(self.initializations)
        )

    def dimensions(self) -> dict[str, Any]:
        return {
            "activation": list(self.activations),
            "optimizer": list(self.optimizers),
            "lr": list(self.learning_rates),
            "init": list(self.initializations),
        }


def default_space(dataset: str | None = None) -> ParameterSpace:
    """Devuelve el espacio de protocolo inicial para ``dataset``.

    Sin argumento (o con ``wine_quality``) conserva exactamente los valores
    históricos y no toca la red ni el disco. Para cualquier otro dataset
    registrado, deriva ``input_dim``/``output_dim`` del CSV en caché mediante
    :func:`prebenchmark.data.dataset_base_config`.
    """
    if dataset is None or dataset == DEFAULT_DATASET:
        base = ExperimentConfig(
            activation=BASELINE_ACTIVATION,
            optimizer=BASELINE_OPTIMIZER,
            lr=BASELINE_LR,
            init=BASELINE_INIT,
        )
    else:
        from . import data  # import diferido: torch/scikit-learn solo si hace falta

        base = data.dataset_base_config(dataset)
    return ParameterSpace(base=base)


def _copy_base(base: ExperimentConfig) -> ExperimentConfig:
    """Copia aproximadamente profunda para que cada configuración generada posea sus especificaciones anidadas."""
    return replace(
        base,
        dataset=replace(base.dataset),
        architecture=replace(
            base.architecture,
            hidden_layers=list(base.architecture.hidden_layers),
        ),
        split=replace(base.split),
        preprocessing=replace(base.preprocessing),
        convergence=replace(base.convergence),
    )


def generate_configs(space: ParameterSpace | None = None) -> list[ExperimentConfig]:
    """Genera el conjunto completo de candidatos como producto cartesiano (sin codificación fija)."""
    active_space = default_space() if space is None else space
    configs: list[ExperimentConfig] = []
    for activation, optimizer, lr, init in product(
        active_space.activations,
        active_space.optimizers,
        active_space.learning_rates,
        active_space.initializations,
    ):
        cfg = _copy_base(active_space.base)
        cfg.activation = activation
        cfg.optimizer = optimizer
        cfg.lr = lr
        cfg.init = init
        configs.append(cfg)
    return configs


def space_key_dims(cfg: ExperimentConfig) -> dict[str, Any]:
    """Devuelve las cuatro dimensiones que definen el espacio de ``cfg``.

    ``class_weight`` y todos los ajustes de protocolo se excluyen por construcción.
    """
    return {
        "activation": cfg.activation,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "init": cfg.init,
    }


def compute_space_key(cfg: ExperimentConfig) -> str:
    """Hash de contenido estable sobre las cuatro dimensiones del espacio.

    Dos configuraciones que difieren solo en ``class_weight`` (o en cualquier campo
    ajeno al espacio) comparten la misma clave de espacio; cualquier cambio en las cuatro
    dimensiones la modifica.
    """
    payload = json.dumps(
        space_key_dims(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"space-v1-{digest}"


def space_keys_unique(configs: Iterable[ExperimentConfig]) -> bool:
    """True cuando cada configuración se asigna a una clave de espacio distinta."""
    keys = [compute_space_key(cfg) for cfg in configs]
    return len(keys) == len(set(keys))
