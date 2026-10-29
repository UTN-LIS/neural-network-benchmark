# Delta for training-runner

## ADDED Requirements

### Requirement: Reproducible Execution Pipeline
The system MUST execute training runs in a reproducible manner using the recorded configuration, with controlled randomness and import-safe execution.

#### Scenario: Deterministic seeding
- GIVEN `random_state=42`
- WHEN the runner initializes
- THEN numpy and torch RNGs MUST be seeded consistently
- AND all relevant stochastic operations MUST use the controlled seed

#### Scenario: Import-safe execution
- GIVEN the runner is used as a library component
- THEN execution MUST NOT trigger module-level side effects on import
- AND dataset loading/training MUST occur only when explicitly invoked

#### Scenario: Use recorded configuration
- GIVEN a persisted configuration
- WHEN executed
- THEN the runner MUST use the exact parameter values from that configuration

### Requirement: Dataset Loading with Caching and Call-Scope Fetch
The system MUST load the wine_quality dataset at call scope with caching, accepting that the UCI dataset may require network download on first run, and never importing legacy `Experimentos/*.py` scripts.

#### Scenario: Load wine_quality on demand
- GIVEN wine_quality dataset is requested
- WHEN loading occurs
- THEN it MUST be loaded at call scope (not import time)
- AND results MUST be cached to avoid repeated network fetches

#### Scenario: Never import legacy scripts
- GIVEN runner code
- WHEN importing dataset/loading logic
- THEN it MUST NOT import any module from `Experimentos/`
- AND it MUST NOT depend on `wine_quality.py` implementation

#### Scenario: Accept first-run download
- GIVEN no cached dataset available
- WHEN loading wine_quality
- THEN network download MAY occur on first run
- AND the system MUST handle it gracefully without hardcoding local paths to legacy copies

### Requirement: Train/Validation/Test Split Implementation
The system MUST create train/validation/test splits according to recorded proportions (default 60/20/20), configurable and recorded.

#### Scenario: Apply default split
- GIVEN default proportions 60/20/20
- WHEN splitting data
- THEN splits MUST respect train=0.6, val=0.2, test=0.2
- AND stratification MUST be applied where appropriate for classification

#### Scenario: Apply custom split
- GIVEN explicitly specified split proportions
- WHEN splitting
- THEN the exact proportions MUST be used
- AND deviations due to rounding MUST be documented or handled consistently

#### Scenario: Record split in metadata
- GIVEN splits are created
- THEN split proportions and sizes MUST be recorded in run metadata

### Requirement: DataLoader with Batch Size 32
The system MUST use a DataLoader with batch size 32 by default, supporting mini-batch training.

#### Scenario: Default batch size
- GIVEN batch_size not specified
- WHEN creating DataLoader
- THEN batch_size MUST be 32

#### Scenario: Use mini-batches
- GIVEN training data larger than batch size
- THEN training MUST iterate over mini-batches via DataLoader
- AND epoch-level metrics MUST be computed over the epoch (aggregated appropriately)

### Requirement: Model Construction and Weight Initialization
The system MUST construct the neural network according to architecture specification and apply initialization (Xavier/Glorot or He/Kaiming) as specified.

#### Scenario: Build architecture from config
- GIVEN architecture specification in configuration
- WHEN model is constructed
- THEN layers/structure MUST match the specified architecture
- AND the existing wine_quality architecture baseline MUST be used as the reference when not overridden

#### Scenario: Apply He/Kaiming initialization
- GIVEN initialization is He/Kaiming
- WHEN weights are initialized
- THEN initialization MUST follow He/Kaiming scheme appropriate for the activation function

#### Scenario: Apply Xavier/Glorot initialization
- GIVEN initialization is Xavier/Glorot
- THEN Xavier/Glorot initialization MUST be applied as specified

### Requirement: Optimizer and Loss Configuration
The system MUST support SGD and Adam optimizers with specified learning rates, and use CrossEntropyLoss as the loss function.

#### Scenario: Use Adam optimizer
- GIVEN optimizer is Adam with lr=0.001
- THEN Adam optimizer MUST be instantiated with the specified learning rate
- AND other parameters MUST remain explicit or documented

#### Scenario: Use SGD optimizer
- GIVEN optimizer is SGD with specified lr
- THEN SGD MUST be used with the recorded learning rate

#### Scenario: Use CrossEntropyLoss
- GIVEN loss function is CrossEntropyLoss
- THEN CrossEntropyLoss MUST be used for training and evaluation

#### Scenario: Weight the training objective by class frequency
- GIVEN preprocessing declares `class_weight="balanced"`
- WHEN the training objective is computed
- THEN CrossEntropyLoss MUST be weighted by the inverse class frequency of the training split
- AND the class weights MUST be derived from the training split only (no test/validation leakage)
- AND the applied weights MUST be recorded in the run metadata

#### Scenario: Evaluate with an unweighted loss
- GIVEN preprocessing declares `class_weight="balanced"`
- WHEN validation or test metrics are computed
- THEN the loss MUST be the UNWEIGHTED cross-entropy of the evaluation set
- AND the recorded `val_loss` and the final `loss` MUST measure the same quantity in every run, regardless of `class_weight`
- AND the weighting MUST NOT change the reported loss scale between runs, so that differences of `loss` remain comparable

### Requirement: Per-Epoch Metric Capture During Training
The system MUST capture training loss and accuracy per epoch; validation loss and accuracy per epoch when validation set exists.

#### Scenario: Capture training metrics per epoch
- GIVEN training proceeds epoch by epoch
- WHEN each epoch completes
- THEN training loss and training accuracy MUST be recorded for that epoch

#### Scenario: Capture validation metrics when available
- GIVEN a validation set exists
- WHEN each epoch completes
- THEN validation loss and validation accuracy MUST be computed and recorded
- AND if no validation set exists, validation metrics MUST be recorded as N/A

#### Scenario: Track epochs executed
- GIVEN max_epochs is 500
- WHEN training completes
- THEN `epochs_executed` (actual number of epochs run) MUST be recorded in metadata
- AND it MUST equal the count of epochs for which metrics were persisted

### Requirement: Convergence Criterion Support
The system MUST support an explicitly declared convergence criterion (default OFF). When enabled, termination must follow the declared criterion and all parameters must be recorded.

#### Scenario: Convergence disabled by default
- GIVEN convergence criterion is disabled
- WHEN training runs
- THEN training MUST run for up to max_epochs
- AND termination MUST be by epoch limit unless error occurs

#### Scenario: Early stopping on validation loss
- GIVEN convergence is enabled with early stopping on validation loss and patience specified
- WHEN validation loss fails to improve for patience epochs
- THEN training MUST stop early
- AND `epochs_executed` MUST reflect the actual stopping epoch
- AND the convergence trigger (reason, metric, patience, min_delta) MUST be recorded in metadata

#### Scenario: All convergence params recorded
- GIVEN any convergence criterion is used
- THEN every parameter value (enabled, metric, patience, min_delta, mode, etc.) MUST appear in configuration and run metadata
- AND no hidden thresholds MUST be applied

### Requirement: Detection of Numerical Issues
The system MUST detect NaN or infinite values during training and record them appropriately.

#### Scenario: Detect NaN in loss
- GIVEN training loss becomes NaN
- THEN the run MUST be marked appropriately (e.g., `FAILED` or `COMPLETED_WITH_WARNINGS` with details)
- AND the epoch where NaN occurred MUST be recorded
- AND the run MUST still be persisted with failure/reason details

#### Scenario: Detect inf values
- GIVEN metrics contain infinite values
- THEN they MUST be detected and recorded as warnings/errors
- AND affected epochs MUST be identified

### Requirement: Final Evaluation on Test Set
The system MUST compute final metrics on the test set (or designated evaluation set as per configuration), including the full metric suite.

#### Scenario: Compute full metric suite on test set
- GIVEN test set exists
- THEN final evaluation MUST compute accuracy, balanced accuracy, precision macro, recall macro, F1 macro, loss, and confusion matrix on test data
- AND each metric MUST be labeled by evaluation set (test)

#### Scenario: Consistent metric computation
- GIVEN metrics are computed
- THEN macro-averaged metrics MUST be computed consistently across classes
- AND balanced accuracy MUST account for class imbalance

### Requirement: Environment and Device Capture
The system MUST capture and record device used, hardware availability, and relevant environment context in run metadata.

#### Scenario: Record device
- GIVEN execution occurs on CPU or GPU
- THEN the actual device used MUST be recorded in metadata

#### Scenario: Record environment context
- GIVEN the run executes
- THEN relevant software/library versions and hardware context MUST be recorded where reliably obtainable
- AND missing information MUST not cause failure

### Requirement: Failure Handling with Partial Persistence
The system MUST handle exceptions gracefully and persist partial results when training fails.

#### Scenario: Persist on exception
- GIVEN an exception occurs mid-training
- THEN the runner MUST catch it, record failure reason and state `FAILED`
- AND persist configuration snapshot, metadata (with timestamps), any epoch history collected so far, and error details
- AND the run MUST receive a unique `run_id`

#### Scenario: No overwrite of prior runs
- GIVEN prior runs exist
- WHEN current run fails
- THEN new `run_id` MUST be used and prior runs preserved

### Requirement: Baseline Reference Definition
The system MUST treat the existing wine_quality configuration (ReLU + Adam + lr 0.001 + He init + CrossEntropyLoss + existing wine_quality architecture) as the baseline reference, with assumptions explicitly recorded.

#### Scenario: Use baseline as default reference
- GIVEN no alternative baseline specified
- THEN baseline MUST be identified as ReLU + Adam + lr 0.001 + He init + CrossEntropyLoss + existing wine_quality architecture
- AND baseline assumptions MUST be recorded in metadata when used as reference

#### Scenario: Record baseline identity
- GIVEN a run is designated as baseline
- THEN its identity as baseline and the rationale MUST be recorded in decision/comparison context
- AND the baseline configuration snapshot MUST be preserved