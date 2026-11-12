"""Modelo de configuración tipado y serializable para experimentos de prebenchmark.

Este módulo define únicamente el *esquema* de configuración. No realiza E/S ni
importa bibliotecas de terceros, por lo que es seguro importarlo en cualquier
contexto (incluida la recolección de tests y contextos de biblioteca).

El modelo separa intencionalmente los parámetros libres (activación, optimizador,
tasa de aprendizaje, inicialización) de los ajustes de protocolo o adyacentes al
protocolo (dataset, arquitectura, partición, preprocesamiento, convergencia), de
modo que el espacio de 36 combinaciones puede generarse de forma explícita en
:mod:`prebenchmark.space`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Registros de nombres -- la única fuente de verdad para la validación.
# Extender el espacio significa agregar un nombre aquí (datos), no reescribir lógica.
# ---------------------------------------------------------------------------

KNOWN_ACTIVATIONS: tuple[str, ...] = ("relu", "sigmoid", "tanh")
KNOWN_OPTIMIZERS: tuple[str, ...] = ("sgd", "adam")
KNOWN_INITIALIZATIONS: tuple[str, ...] = ("xavier", "he")
KNOWN_LOSSES: tuple[str, ...] = ("cross_entropy",)
KNOWN_SCALERS: tuple[str, ...] = ("standard",)
KNOWN_RUN_TYPES: tuple[str, ...] = ("prebenchmark", "benchmark")

# Registro de datasets soportados. ``prebenchmark.data`` es la fuente de verdad
# de los loaders; esta tupla existe para que la CLI y la validación puedan
# ofrecer las opciones sin importar torch/scikit-learn. Un test verifica que
# ambas listas coincidan.
DEFAULT_DATASET = "wine_quality"
KNOWN_DATASETS: tuple[str, ...] = (
    "wine_quality",
    "satimage",
    "vehicle",
    "waveform_5000",
    "segment",
    "digits",
)
KNOWN_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
KNOWN_CONVERGENCE_METRICS: tuple[str, ...] = ("val_loss", "train_loss")
KNOWN_CONVERGENCE_MODES: tuple[str, ...] = ("min", "max")

# ---------------------------------------------------------------------------
# Valores por defecto del protocolo (requisitos: partición 60/20/20, lote 32, 500 épocas, 42).
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 500
DEFAULT_SEED = 42
DEFAULT_DEVICE = "auto"
DEFAULT_RUN_TYPE = "prebenchmark"

# ---------------------------------------------------------------------------
# Referencia de línea base: la única configuración que se ejecuta en este repositorio
# (Experimentos/wine_quality.py): ReLU + Adam + lr 0.001 + He + CrossEntropyLoss.
# ---------------------------------------------------------------------------

BASELINE_ACTIVATION = "relu"
BASELINE_OPTIMIZER = "adam"
BASELINE_LR = 0.001
BASELINE_INIT = "he"

# ---------------------------------------------------------------------------
# Criterio de convergencia por defecto documentado, aplicado SOLO cuando la
# convergencia está habilitada y un campo queda sin especificar. Son valores
# explícitos y registrados -- nunca un umbral oculto: el criterio resuelto se
# escribe en los metadatos de la ejecución, de modo que la regla de parada exacta
# utilizada siempre es recuperable.
# ---------------------------------------------------------------------------

DEFAULT_CONVERGENCE_METRIC = "val_loss"
DEFAULT_CONVERGENCE_PATIENCE = 20
DEFAULT_CONVERGENCE_MIN_DELTA = 1e-4
DEFAULT_CONVERGENCE_MODE = "min"


@dataclass
class DatasetSpec:
    """Identidad del dataset y de la fuente/versión."""

    name: str = "wine_quality"
    source: str = "UCI winequality-white.csv"
    version: str | None = None
    target: str = "quality"
    n_features: int = 11

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSpec:
        return cls(
            name=data.get("name", "wine_quality"),
            source=data.get("source", "UCI winequality-white.csv"),
            version=data.get("version"),
            target=data.get("target", "quality"),
            n_features=data.get("n_features", 11),
        )


@dataclass
class ArchitectureSpec:
    """Especificación de la arquitectura de la red."""

    name: str = "wine_quality_mlp"
    input_dim: int = 11
    hidden_layers: list[int] = field(default_factory=lambda: [32, 16])
    output_dim: int = 7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureSpec:
        return cls(
            name=data.get("name", "wine_quality_mlp"),
            input_dim=data.get("input_dim", 11),
            hidden_layers=list(data.get("hidden_layers", [32, 16])),
            output_dim=data.get("output_dim", 7),
        )


@dataclass
class SplitSpec:
    """Proporciones de la partición de entrenamiento/validación/prueba."""

    train: float = DEFAULT_TRAIN_RATIO
    val: float = DEFAULT_VAL_RATIO
    test: float = DEFAULT_TEST_RATIO
    stratify: bool = True

    def proportions(self) -> tuple[float, float, float]:
        return (self.train, self.val, self.test)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitSpec:
        return cls(
            train=data.get("train", DEFAULT_TRAIN_RATIO),
            val=data.get("val", DEFAULT_VAL_RATIO),
            test=data.get("test", DEFAULT_TEST_RATIO),
            stratify=data.get("stratify", True),
        )


@dataclass
class PreprocessingSpec:
    """Detalles del preprocesamiento, incluido el ``class_weight`` registrable.

    ``class_weight`` es un campo registrable pero intencionalmente NO forma parte de
    la clave del espacio de 4 dimensiones (véase :func:`prebenchmark.space.compute_space_key`).
    """

    scaler: str = "standard"
    class_weight: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingSpec:
        return cls(
            scaler=data.get("scaler", "standard"),
            class_weight=data.get("class_weight"),
        )


@dataclass
class ConvergenceSpec:
    """Criterio de convergencia explícito. Por defecto está DESACTIVADO (sin umbrales ocultos)."""

    enabled: bool = False
    metric: str | None = None
    patience: int | None = None
    min_delta: float | None = None
    mode: str = "min"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceSpec:
        return cls(
            enabled=data.get("enabled", False),
            metric=data.get("metric"),
            patience=data.get("patience"),
            min_delta=data.get("min_delta"),
            mode=data.get("mode", "min"),
        )


def resolve_convergence(spec: ConvergenceSpec) -> ConvergenceSpec:
    """Devuelve ``spec`` con cada campo sin especificar completado con el valor por defecto documentado.

    Los valores por defecto se aplican campo por campo (``metric``, ``patience``, ``min_delta``,
    ``mode``). Esto nunca modifica el :class:`ConvergenceSpec` almacenado; solo
    produce el criterio concreto que el ejecutor aplicará cuando la convergencia esté
    habilitada, de modo que los valores resueltos puedan registrarse en los metadatos de la ejecución.
    """
    return ConvergenceSpec(
        enabled=spec.enabled,
        metric=spec.metric if spec.metric is not None else DEFAULT_CONVERGENCE_METRIC,
        patience=spec.patience if spec.patience is not None else DEFAULT_CONVERGENCE_PATIENCE,
        min_delta=(spec.min_delta if spec.min_delta is not None else DEFAULT_CONVERGENCE_MIN_DELTA),
        mode=spec.mode if spec.mode else DEFAULT_CONVERGENCE_MODE,
    )


@dataclass
class ExperimentConfig:
    """Configuración completa del experimento.

    Los valores por defecto reflejan la línea base documentada, de modo que ``ExperimentConfig()``
    es una configuración ejecutable y VALID.
    """

    run_type: str = DEFAULT_RUN_TYPE
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    architecture: ArchitectureSpec = field(default_factory=ArchitectureSpec)
    activation: str = BASELINE_ACTIVATION
    optimizer: str = BASELINE_OPTIMIZER
    lr: float = BASELINE_LR
    init: str = BASELINE_INIT
    loss: str = "cross_entropy"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    seed: int = DEFAULT_SEED
    split: SplitSpec = field(default_factory=SplitSpec)
    preprocessing: PreprocessingSpec = field(default_factory=PreprocessingSpec)
    convergence: ConvergenceSpec = field(default_factory=ConvergenceSpec)
    device: str = DEFAULT_DEVICE

    def to_dict(self) -> dict[str, Any]:
        """Devuelve un dict plano, con orden estable y serializable a JSON.

        El orden de los campos sigue la definición del dataclass, lo que mantiene
        la salida serializada canónica (importante para la identidad por hash de contenido en etapas posteriores).
        """
        return {
            "run_type": self.run_type,
            "dataset": self.dataset.to_dict(),
            "architecture": self.architecture.to_dict(),
            "activation": self.activation,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "init": self.init,
            "loss": self.loss,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "seed": self.seed,
            "split": self.split.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "convergence": self.convergence.to_dict(),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Reconstruye una configuración a partir de la salida de :meth:`to_dict`."""
        return cls(
            run_type=data.get("run_type", DEFAULT_RUN_TYPE),
            dataset=DatasetSpec.from_dict(data.get("dataset", {})),
            architecture=ArchitectureSpec.from_dict(data.get("architecture", {})),
            activation=data.get("activation", BASELINE_ACTIVATION),
            optimizer=data.get("optimizer", BASELINE_OPTIMIZER),
            lr=data.get("lr", BASELINE_LR),
            init=data.get("init", BASELINE_INIT),
            loss=data.get("loss", "cross_entropy"),
            batch_size=data.get("batch_size", DEFAULT_BATCH_SIZE),
            max_epochs=data.get("max_epochs", DEFAULT_MAX_EPOCHS),
            seed=data.get("seed", DEFAULT_SEED),
            split=SplitSpec.from_dict(data.get("split", {})),
            preprocessing=PreprocessingSpec.from_dict(data.get("preprocessing", {})),
            convergence=ConvergenceSpec.from_dict(data.get("convergence", {})),
            device=data.get("device", DEFAULT_DEVICE),
        )


EXPLORATORY_SEEDS: tuple[int, ...] = (42, 123, 7, 2025, 999)
