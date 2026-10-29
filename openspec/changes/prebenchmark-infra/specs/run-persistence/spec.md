# Delta for run-persistence

## ADDED Requirements

### Requirement: Persistent Run Identity
The system MUST assign each execution a persistent, unique run identifier independent of outcome, and must maintain stable configuration identity.

#### Scenario: Generate unique run IDs
- GIVEN a new execution is started
- THEN a unique `run_id` MUST be generated
- AND it MUST persist with the run record regardless of success or failure

#### Scenario: Maintain config identity
- GIVEN a configuration
- THEN a stable `config_id` MUST be derived from its identifying parameters
- AND the same configuration parameters MUST produce the same `config_id` across runs

#### Scenario: Distinguish runs from same config
- GIVEN the same configuration executed multiple times (different seeds or repeated runs)
- THEN each execution MUST receive a distinct `run_id`
- AND all runs MUST reference the same `config_id`

### Requirement: No Overwrite of Existing Runs
The system MUST preserve historical runs and never overwrite previously persisted execution records.

#### Scenario: Prevent overwriting on re-execution
- GIVEN a run with a specific `run_id` is already persisted
- WHEN attempting to save results for that same `run_id`
- THEN the system MUST NOT overwrite existing files/records
- AND it MUST either generate a new `run_id` or fail explicitly

#### Scenario: Historical preservation across multiple runs
- GIVEN the same configuration is run multiple times
- THEN each run MUST be persisted in its own location/container keyed by `run_id`
- AND all prior run records MUST remain intact

### Requirement: Complete Run Metadata Recording
The system MUST record complete metadata for each run including timestamps, execution context, environment, device, and configuration snapshot.

#### Scenario: Record timestamps
- GIVEN a run starts and ends
- THEN `start_time` and `end_time` (or equivalent timing) MUST be recorded in run metadata

#### Scenario: Record execution context
- GIVEN a run executes
- THEN metadata MUST include dataset and version/source, seed (`random_state`), architecture, optimizer, learning rate, initialization, batch size, max epochs, actual epochs executed, loss function, split proportions, convergence criterion configuration (if any), and preprocessing details

#### Scenario: Record environment and device
- GIVEN a run executes
- THEN metadata MUST record device used (CPU/GPU), hardware context if available, and relevant software/library context
- AND the information recorded MUST reflect actual execution conditions

#### Scenario: Snapshot full configuration
- GIVEN a run is persisted
- THEN the complete configuration as used MUST be stored in the run record
- AND it MUST be sufficient to reproduce the run

### Requirement: Per-Epoch Metrics Persistence
The system MUST persist metrics collected during training on a per-epoch basis, including training and validation metrics when available.

#### Scenario: Record per-epoch history
- GIVEN training runs for N epochs
- WHEN epochs complete
- THEN for each epoch the system MUST persist `epoch`, training loss, and training accuracy (if computed)
- AND validation loss and validation accuracy MUST be persisted when a validation set exists; otherwise recorded as N/A

#### Scenario: Include timing if available
- GIVEN epoch-level timing is measurable consistently
- THEN epoch duration/time MUST be recorded when available

#### Scenario: Epoch history remains immutable
- GIVEN epoch metrics for a run are persisted
- THEN they MUST NOT be modified after persistence

### Requirement: Final Metrics and Confusion Matrix Persistence
The system MUST persist final evaluation metrics computed on the appropriate evaluation set, including confusion matrix and macro metrics.

#### Scenario: Record final metrics
- GIVEN evaluation completes
- THEN final metrics MUST include loss, accuracy, balanced accuracy, precision macro, recall macro (sensitivity macro), F1-score macro
- AND the evaluation set (train/val/test) for each reported metric MUST be clearly identified

#### Scenario: Preserve confusion matrix
- GIVEN evaluation produces a confusion matrix
- THEN the confusion matrix MUST be persisted in sufficient detail to identify errors per class

#### Scenario: No cross-set mixing in comparison views
- GIVEN metrics are stored
- THEN the storage structure MUST distinguish metrics by evaluation set to prevent mixing train/val/test values in comparisons

### Requirement: Failed Run Persistence
The system MUST persist failed runs with full context, never overwriting prior runs, and record failure reason and state.

#### Scenario: Persist on failure
- GIVEN a run fails due to error
- THEN the run MUST still be persisted with state `FAILED`
- AND failure reason, error details, and stack trace/context MUST be recorded
- AND partial artifacts (configuration, metadata, epoch history collected so far) MUST be preserved where available

#### Scenario: Do not overwrite existing
- GIVEN a prior successful run exists
- WHEN a subsequent run with same logical intent fails
- THEN the failed run MUST be stored under a different `run_id`
- AND the prior successful run MUST remain unchanged

#### Scenario: Distinguish failure states
- GIVEN execution completes with issues
- THEN the system MUST distinguish between `FAILED` (error/exception), `COMPLETED`, and `COMPLETED_WITH_WARNINGS` as appropriate
- AND warnings encountered MUST be recorded in the run record

### Requirement: Structured Storage Separation
The system MUST separate configurations, run metadata, epoch metrics, final metrics, warnings/errors, and decision records in a structured, queryable form while avoiding unnecessary duplication.

#### Scenario: Separate concerns
- GIVEN run persistence
- THEN configurations, metadata, epoch metrics, and final metrics MUST be stored as distinct components
- AND references by `config_id` and `run_id` MUST link them without duplicating full configuration in every metric row unnecessarily

#### Scenario: Hybrid on-disk layout
- GIVEN the recommended hybrid layout
- THEN each run MUST be stored under its own directory keyed by `run_id` containing `metadata.json` (and/or structured metadata) and `epochs.csv`
- AND aggregate tables/exports MAY be produced separately for analysis
- AND the layout MUST be transparent and portable for pandas consumption

### Requirement: Export Capability from Persistent Store
The system MUST support exporting persisted results to CSV and equivalent formats for external analysis.

#### Scenario: Export master run table
- GIVEN persisted runs exist
- WHEN exporting master table
- THEN all runs with their key metadata and final metrics MUST be exportable to CSV

#### Scenario: Export epoch history
- GIVEN epoch metrics exist
- WHEN exporting
- THEN epoch history MUST be exportable to CSV in a flat form suitable for plotting

#### Scenario: Preserve data integrity on export
- GIVEN exports are generated
- THEN exported data MUST reflect persisted values without transformation that loses information
- AND field names MUST be clear and unambiguous

### Requirement: Historical Query Support
The system MUST allow retrieval of runs by filters (config_id, dataset, architecture, activation, optimizer, learning rate, initialization, batch size, seed, run type, state) without relying on in-memory process state.

#### Scenario: Filter by configuration dimensions
- GIVEN multiple persisted runs
- WHEN filtering by activation/optimizer/lr/init
- THEN matching runs MUST be returned

#### Scenario: Filter by dataset and architecture
- GIVEN runs across datasets/architectures
- WHEN filtered
- THEN only runs matching criteria MUST be returned
- AND results MUST include run_id, config_id, state, timestamps, and key metrics

#### Scenario: Retrieve full run record
- GIVEN a `run_id`
- WHEN requested
- THEN the system MUST return complete metadata, full configuration snapshot, epoch history, final metrics, warnings/errors, and any decision records associated with that run