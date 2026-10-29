# Delta for results-export

## ADDED Requirements

### Requirement: Master Runs Table Export
The system MUST export a master table containing all runs with key metadata and final metrics.

#### Scenario: Export complete run list
- GIVEN persisted runs exist (including FAILED runs)
- THEN master table MUST include run_id, config_id, dataset, architecture, activation, optimizer, lr, init, batch size, seed, split proportions, max epochs, epochs executed, execution state, timestamps, device, final metrics, and warnings summary

#### Scenario: Include all runs regardless of state
- GIVEN some runs FAILED
- THEN they MUST appear in master table with state and failure reason
- AND performance metrics MAY be empty/marked N/A for failed runs

#### Scenario: CSV format
- GIVEN export
- THEN master table MUST be exportable as CSV with clear column names

### Requirement: Baseline Comparison Table Export
The system MUST export baseline comparison table showing deltas vs selected baseline with comparability information.

#### Scenario: Include baseline reference
- GIVEN baseline is selected
- THEN exported table MUST identify baseline run_id/config_id
- AND include rationale for baseline selection

#### Scenario: Include deltas and flags
- GIVEN comparison
- THEN table MUST include Δ values for comparable metrics, evaluation set labels, comparability flags (comparable/non-comparable), controlled/multi-variable flag, and warnings

#### Scenario: Handle non-comparable entries
- GIVEN non-comparable runs
- THEN deltas MUST be omitted or marked N/A
- AND comparability reason MUST be stated

### Requirement: Aggregated Repetitions Table Export
The system MUST export aggregated statistics per repetition group.

#### Scenario: Include group statistics
- GIVEN repetition groups exist
- THEN aggregate table MUST include group_id/config_id, member counts (total/valid/failed), configuration summary, mean/std/min/max for each metric, insufficient-repetitions flag, and any group-level notes

#### Scenario: Distinguish valid vs failed contributions
- GIVEN failures in group
- THEN failed count and proportion MUST be explicit
- AND only valid runs contribute to aggregates

### Requirement: Per-Epoch Metrics Export
The system MUST export per-epoch history for runs in a flat, analyzable format.

#### Scenario: Export epochs CSV
- GIVEN epoch metrics exist
- THEN epochs table MUST be exportable as CSV with columns including run_id, config_id, epoch, training loss, training accuracy, validation loss, validation accuracy (or N/A), epoch time if available

#### Scenario: Support filtering
- GIVEN large history
- THEN export MAY be filtered by run_id or config_id as requested

#### Scenario: Preserve epoch order
- GIVEN export
- THEN epochs MUST be ordered by run_id then epoch number

### Requirement: Decision Log Export
The system MUST export the decision log with full context.

#### Scenario: Complete decision export
- GIVEN decision entries exist
- THEN export MUST include decision_id, entity references (run_id/config_id/group_id), decision type, exploratory label if applicable, justification, actor, timestamp, and any supporting evidence

#### Scenario: Append-only history preserved
- GIVEN export
- THEN all decisions in chronological order MUST be included

### Requirement: Configurations Export
The system MUST export configuration definitions (including validation states) for reproducibility.

#### Scenario: Export all configurations
- GIVEN configurations exist
- THEN table MUST include config_id, full parameter values, preprocessing details (including class_weight), convergence configuration, validation state (VALID/WARNING/INVALID), validation rationale, and timestamps

#### Scenario: Support reproducibility
- GIVEN exported configuration
- THEN it MUST contain sufficient information to recreate the exact configuration

### Requirement: Error/Warning Summary Export
The system MUST export consolidated errors, warnings, and runs pending review.

#### Scenario: Export warnings/errors list
- GIVEN runs have warnings or failures
- THEN summary MUST list run_id, state, warning/error messages with context, categories, and epochs if applicable

#### Scenario: Export pending review
- GIVEN runs with exploratory label `PENDING_REVIEW`
- THEN they MUST be included in a pending-review summary export

### Requirement: Export Naming and Organization
The system MUST produce exports with clear naming and organize outputs for transparency.

#### Scenario: Consistent naming
- GIVEN exports are generated
- THEN filenames MUST be descriptive and include context (e.g., timestamp or export type)
- AND be portable across filesystems

#### Scenario: Organize by type
- GIVEN multiple export types
- THEN each type MUST be clearly identified (master, baseline_comparison, aggregates, epochs, decisions, configs, warnings_summary)

### Requirement: Data Integrity in Exports
The system MUST ensure exported data accurately reflects persisted state without loss or corruption.

#### Scenario: Lossless export
- GIVEN persisted values
- THEN exports MUST preserve numeric precision and text content
- AND no fields MUST be silently dropped

#### Scenario: N/A handling
- GIVEN missing values (e.g., validation metrics when no val set)
- THEN exports MUST represent them as explicit N/A/NaN consistently

### Requirement: Export from Results Directory
The system MUST write exports to the `results/` output directory structure as appropriate, supporting the project's convention.

#### Scenario: Use results directory
- GIVEN exports are generated
- THEN they MUST be written under a `results/` directory tree
- AND the layout MUST be organized by export type/timestamp for discoverability

#### Scenario: Respect gitignore convention
- GIVEN `results/` contains generated run artifacts/exports
- THEN the system design MUST align with the convention that `results/` output is not committed (per proposal: `results/` git-ignored) — exports may be generated locally for analysis

### Requirement: Optional Visualizations
The system MAY generate plots when useful, without making visualization mandatory.

#### Scenario: Optional plotting support
- GIVEN visualization dependencies are available
- THEN plots for epoch evolution, metric comparisons, variability may be generated
- AND visualization MUST NOT be required for core export functionality

#### Scenario: Non-blocking if unavailable
- GIVEN plotting libraries unavailable
- THEN exports MUST still complete successfully