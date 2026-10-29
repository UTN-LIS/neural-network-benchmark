# Delta for experiment-config-space

## ADDED Requirements

### Requirement: Config Model Definition
The system MUST define a typed, serializable configuration model capturing all parameters that affect execution.

#### Scenario: Store complete configuration
- GIVEN an experiment configuration is defined
- WHEN the configuration is serialized to the persistent store
- THEN the complete parameter set MUST be preserved without truncation

#### Scenario: Identify configuration uniquely
- GIVEN two configurations with identical parameter values
- THEN they MUST yield the same `config_id`
- AND given configurations differing in any parameter value
- THEN they MUST yield different `config_id`s

### Requirement: Explicit Parameter Space Generation
The system MUST generate the full candidate set from an explicit parameter space specification without hardcoding individual experiments.

#### Scenario: Generate 36 combinations
- GIVEN activation set {ReLU, Sigmoid, Tanh}
- AND optimizer set {SGD, Adam}
- AND learning rate set {0.001, 0.01, 0.1}
- AND initialization set {Xavier/Glorot, He/Kaiming}
- WHEN candidate configurations are generated
- THEN the system MUST produce exactly 36 combinations
- AND each combination MUST be uniquely identifiable by its `config_id`

#### Scenario: Maintain fixed combinatorial space
- GIVEN the 36-combination space is generated
- WHEN combinations are enumerated
- THEN no combination MAY be dropped silently
- AND no additional combinations MAY be injected without explicit space modification

#### Scenario: Allow extensibility
- GIVEN an extended activation set beyond {ReLU, Sigmoid, Tanh}
- WHEN the space is regenerated from the extended specification
- THEN the system MUST accommodate new values without modifying core validation logic

### Requirement: Class Weight Configuration Separation
The system MUST treat class balancing via `class_weight` as a recordable preprocessing configuration field that is NOT part of the 36-combination parameter space.

#### Scenario: Record class_weight in config
- GIVEN a configuration specifies preprocessing with `class_weight` (e.g., 'balanced' or None)
- WHEN the configuration is persisted
- THEN the `class_weight` value MUST be recorded in the configuration's preprocessing section

#### Scenario: Exclude class_weight from space key
- GIVEN two configurations identical in all 4 space dimensions (act/opt/lr/init)
- AND differing only in `class_weight` value
- WHEN comparing for space membership
- THEN they MUST be considered the same configuration for the 36-space
- AND share the same space-derived `config_id` characteristics

### Requirement: Configuration Pre-Validation States
The system MUST classify each candidate configuration into exactly one of: `VALID`, `WARNING`, or `INVALID`, with explicit rationale recorded for non-VALID states.

#### Scenario: Mark clearly valid as VALID
- GIVEN a configuration with no known technical incompatibilities
- WHEN validation is performed
- THEN its state MUST be `VALID`
- AND no warning rationale is required

#### Scenario: Mark risky but executable as WARNING
- GIVEN a configuration that is technically valid but presents known risk (e.g., high learning rate with certain optimizer/activation combinations)
- WHEN validation is performed
- THEN its state MUST be `WARNING`
- AND the validation result MUST include a non-empty explanation of the warning

#### Scenario: WARNING configurations remain executable
- GIVEN a configuration in `WARNING` state
- WHEN execution is attempted
- THEN the system MUST permit execution
- AND the warning rationale MUST be persisted with the run

#### Scenario: Mark incompatible as INVALID
- GIVEN a configuration that violates a defined technical or methodological constraint
- WHEN validation is performed
- THEN its state MUST be `INVALID`
- AND the validation result MUST include the explicit reason for rejection
- AND the system MUST prevent execution of configurations marked `INVALID`

#### Scenario: No arbitrary automatic rejections
- GIVEN combinations such as Adam with learning rate 0.1 or non-standard activation-initialization pairings
- WHEN validated against the defined constraints
- THEN they MUST NOT be automatically classified as `INVALID` solely due to convention
- AND high-risk cases MUST be classified as `WARNING` if technically executable

### Requirement: Convergence Criterion Declaration
The system MUST require a convergence criterion to be explicitly declared in configuration, with a documented default that is OFF for the fixed-epoch protocol.

#### Scenario: Default convergence disabled
- GIVEN no convergence criterion is specified in configuration
- WHEN creating a default configuration
- THEN convergence criterion MUST be set to disabled (OFF)
- AND `max_epochs` MUST remain the primary termination control (default 500)

#### Scenario: Record convergence configuration
- GIVEN a configuration specifies convergence settings (e.g., early stopping with validation loss, patience, min_delta, enabled flag)
- WHEN persisted
- THEN all convergence parameters MUST be recorded verbatim in the configuration
- AND the enabled state MUST be explicitly recorded

#### Scenario: Convergence parameters never hidden
- GIVEN a convergence criterion is active during a run
- WHEN run metadata is persisted
- THEN every parameter value used by the convergence logic MUST appear in the run's configuration and metadata
- AND no implicit or hardcoded thresholds outside the declared configuration MAY be used

### Requirement: Configurability of Split and Training Parameters
The system MUST support configurable train/validation/test split proportions with defaults of 60/20/20, and configurable batch size (default 32) and max epochs (default 500), all recorded in configuration.

#### Scenario: Default split proportions
- GIVEN a configuration without explicit split proportions
- WHEN defaults are applied
- THEN train/val/test split MUST default to 60/20/20
- AND these proportions MUST be recorded in the configuration

#### Scenario: Record custom split
- GIVEN split proportions are explicitly specified (e.g., train=0.6, val=0.2, test=0.2)
- WHEN persisted
- THEN the exact proportions MUST be recorded
- AND the split logic MUST use the recorded proportions

#### Scenario: Default batch size
- GIVEN no batch size specified
- WHEN defaults are applied
- THEN `batch_size` MUST default to 32
- AND be recorded in configuration

#### Scenario: Default max epochs
- GIVEN no max_epochs specified
- WHEN defaults are applied
- THEN `max_epochs` MUST default to 500
- AND be recorded in configuration

### Requirement: Reproducibility Control
The system MUST ensure random_state is controlled for all relevant stochastic operations with a protocol default of 42.

#### Scenario: Record seed
- GIVEN a configuration specifies `random_state`
- WHEN persisted
- THEN the seed value MUST be recorded (protocol default 42)

#### Scenario: Apply seed consistently
- GIVEN `random_state=42`
- WHEN initializing stochastic components
- THEN numpy and torch RNGs MUST be seeded to cover operations used by training
- AND the seeding scope MUST be recorded in run metadata
- AND the seed MUST NOT be silently changed between related runs

### Requirement: Dataset and Architecture Reference Recording
The system MUST record dataset identity (including name and version/source identification), architecture specification, loss function, and preprocessing details as part of the configuration.

#### Scenario: Record dataset metadata
- GIVEN dataset is wine_quality for the protocol
- WHEN configuration is persisted
- THEN dataset name and version/source identification MUST be recorded

#### Scenario: Record architecture and loss
- GIVEN architecture and loss function are specified
- WHEN persisted
- THEN architecture details (layers/neurons as applicable) and loss function name MUST be recorded
- AND preprocessing details including `class_weight` (when applicable) MUST be recorded
