"""prebenchmark: infraestructura de registro y comparación experimental.

Seguro para importar: este paquete no realiza E/S ni dispara entrenamiento al importarse.
"""

from __future__ import annotations

from .config import (
    ArchitectureSpec,
    ConvergenceSpec,
    DatasetSpec,
    ExperimentConfig,
    PreprocessingSpec,
    SplitSpec,
    resolve_convergence,
)
from .space import (
    DEFAULT_VALIDATION_RULES,
    HIGH_LR_THRESHOLD,
    INVALID,
    SPACE_DIMENSIONS,
    VALID,
    WARNING,
    ParameterSpace,
    ValidationResult,
    ValidationRule,
    compute_space_key,
    default_space,
    generate_configs,
    space_key_dims,
    space_keys_unique,
    validate_config,
)
from .identity import (
    compute_config_id,
    new_decision_id,
    new_experiment_id,
    new_run_id,
)
from .store import ResultsStore
from .compare import (
    BASELINE_ACTIVATION,
    BASELINE_INIT,
    BASELINE_LR,
    BASELINE_OPTIMIZER,
    COMPARISON_TYPES,
    DELTA_METRICS,
    RELATIVE_BASIS,
    BaselineSelection,
    ComparabilityResult,
    ComparisonRow,
    ComparisonTable,
    check_comparability,
    compare_to_baseline,
    documented_baseline_config,
    filter_runs,
    select_baseline,
)
from .aggregate import (
    AGGREGATE_METRICS,
    DEFAULT_MIN_REPETITIONS,
    GROUPING_MODES,
    MetricStatistics,
    RepetitionGroup,
    RunGroup,
    RunMember,
    aggregate_group,
    aggregate_records,
    aggregate_runs,
    group_runs,
)
from .decisions import (
    DECISION_TYPES,
    EXPLORATORY_LABELS,
    TECHNICAL_STATES,
    decision_to_record,
    decisions_to_records,
    is_exploratory_label,
    is_technical_state,
    record_decision,
)
from .export import (
    AGGREGATE_COLUMNS,
    COMPARISON_COLUMNS,
    CONFIG_COLUMNS,
    DECISION_COLUMNS,
    EPOCH_COLUMNS,
    MASTER_COLUMNS,
    NA_MARKER,
    WARNINGS_COLUMNS,
    ExportResult,
    export_all,
    persist_baseline,
)

__all__ = [
    # configuración
    "ExperimentConfig",
    "DatasetSpec",
    "ArchitectureSpec",
    "SplitSpec",
    "PreprocessingSpec",
    "ConvergenceSpec",
    "resolve_convergence",
    # espacio
    "ParameterSpace",
    "ValidationRule",
    "ValidationResult",
    "VALID",
    "WARNING",
    "INVALID",
    "SPACE_DIMENSIONS",
    "HIGH_LR_THRESHOLD",
    "DEFAULT_VALIDATION_RULES",
    "default_space",
    "generate_configs",
    "validate_config",
    "space_key_dims",
    "compute_space_key",
    "space_keys_unique",
    # identidad
    "compute_config_id",
    "new_run_id",
    "new_experiment_id",
    "new_decision_id",
    # almacén
    "ResultsStore",
    # comparación
    "BASELINE_ACTIVATION",
    "BASELINE_INIT",
    "BASELINE_LR",
    "BASELINE_OPTIMIZER",
    "COMPARISON_TYPES",
    "DELTA_METRICS",
    "RELATIVE_BASIS",
    "BaselineSelection",
    "ComparabilityResult",
    "ComparisonRow",
    "ComparisonTable",
    "check_comparability",
    "compare_to_baseline",
    "documented_baseline_config",
    "filter_runs",
    "select_baseline",
    # agregación
    "AGGREGATE_METRICS",
    "DEFAULT_MIN_REPETITIONS",
    "GROUPING_MODES",
    "MetricStatistics",
    "RepetitionGroup",
    "RunGroup",
    "RunMember",
    "aggregate_group",
    "aggregate_records",
    "aggregate_runs",
    "group_runs",
    # decisiones
    "DECISION_TYPES",
    "EXPLORATORY_LABELS",
    "TECHNICAL_STATES",
    "decision_to_record",
    "decisions_to_records",
    "is_exploratory_label",
    "is_technical_state",
    "record_decision",
    # exportación
    "NA_MARKER",
    "ExportResult",
    "export_all",
    "persist_baseline",
    "MASTER_COLUMNS",
    "COMPARISON_COLUMNS",
    "AGGREGATE_COLUMNS",
    "CONFIG_COLUMNS",
    "DECISION_COLUMNS",
    "EPOCH_COLUMNS",
    "WARNINGS_COLUMNS",
]

from . import seeds
from .seeds import EXPLORATORY_SEEDS, SEEDS_5

