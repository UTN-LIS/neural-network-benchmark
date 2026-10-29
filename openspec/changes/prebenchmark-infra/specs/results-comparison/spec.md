# Delta for results-comparison

## ADDED Requirements

### Requirement: Baseline Comparison Against Reference
The system MUST support comparing runs against a selected baseline reference and compute metric deltas.

#### Scenario: Select baseline explicitly
- GIVEN multiple runs exist
- WHEN selecting a baseline
- THEN the baseline configuration/run MUST be explicitly identified
- AND the selection rationale MUST be recorded

#### Scenario: Compute absolute deltas
- GIVEN baseline run B and candidate run C with comparable metrics
- THEN system MUST compute Δ for accuracy, balanced accuracy, precision macro, recall macro, F1 macro, loss, and training time
- AND deltas MUST be computed on the same evaluation set

#### Scenario: Support relative deltas when meaningful
- GIVEN denominators are valid and non-zero
- WHEN computing relative differences
- THEN relative deltas MAY be included
- AND the basis for relative calculation MUST be explicit

### Requirement: Comparability Guards
The system MUST enforce comparability guards to prevent invalid cross-comparisons.

#### Scenario: Warn on different datasets
- GIVEN baseline and candidate use different datasets
- WHEN comparison is attempted
- THEN the system MUST warn that results are not directly comparable
- AND MUST either block or clearly mark the comparison as non-comparable

#### Scenario: Warn on different partitions/protocols
- GIVEN different split proportions, different evaluation protocol, or different max epochs/convergence settings that affect comparability
- WHEN comparing
- THEN the system MUST detect and warn about protocol differences

#### Scenario: Warn on different architectures
- GIVEN different network architectures
- WHEN comparing
- THEN comparability warning MUST be issued
- AND cross-architecture comparisons MUST be clearly labeled as non-comparable

#### Scenario: Distinguish controlled vs multi-variable comparisons
- GIVEN comparison between runs differing by exactly one variable (e.g., only learning rate)
- THEN it MUST be classified as a controlled comparison
- AND when runs differ by multiple variables simultaneously
- THEN it MUST be classified as multi-variable comparison
- AND this distinction MUST be explicitly presented in comparison output

#### Scenario: Treat class_weight as a varied variable
- GIVEN two runs that differ only in `preprocessing.class_weight`
- THEN the comparison MUST be classified as controlled (one variable: class_weight)
- AND it MUST NOT be classified as identical
- AND the varied dimension MUST be reported as `class_weight`
- AND because `class_weight` does not split the 36-combination space, it MUST NOT alter the space key of the configuration

### Requirement: No Single Composite Weighted Score
The system MUST NOT compute a single weighted composite score combining accuracy, F1, speed, and other metrics with arbitrary weights.

#### Scenario: Preserve full metric profile
- GIVEN comparison results
- THEN all relevant metrics MUST be preserved individually
- AND no aggregate weighted score MAY be computed or presented as the sole ranking criterion

#### Scenario: Present trade-offs explicitly
- GIVEN a configuration improves some metrics while degrading others
- THEN the comparison MUST present all changes
- AND MUST NOT automatically conclude better/worse overall
- AND trade-offs MUST be visible

### Requirement: Comparable Evaluation Sets
The system MUST ensure comparisons only mix metrics from the same evaluation set.

#### Scenario: Compare test-set metrics together
- GIVEN baseline and candidate
- THEN test-set metrics MUST only be compared to test-set metrics
- AND train/val metrics MUST never be mixed into test comparisons

#### Scenario: Label evaluation set
- GIVEN each metric in comparison
- THEN the evaluation set (train/val/test) MUST be clearly labeled

### Requirement: Failed Run Handling in Comparisons
The system MUST handle failed runs appropriately in comparisons, never silently excluding them without explanation.

#### Scenario: Exclude failed runs with rationale
- GIVEN a run in FAILED state
- WHEN included in comparison context
- THEN it MUST be marked as non-comparable for performance metrics
- AND the failure reason MUST be displayed
- AND it MUST NOT contribute computed deltas as if successful

#### Scenario: Preserve failed run records
- GIVEN failed runs exist
- THEN comparisons MUST still list them
- AND their state and failure details MUST remain visible

### Requirement: Comparison Filtering
The system MUST support filtering runs by dataset, architecture, activation, optimizer, learning rate, initialization, batch size, seed, run type, execution state for valid comparison sets.

#### Scenario: Filter to comparable set
- GIVEN a large set of runs
- WHEN filtering
- THEN only runs matching filter criteria form the candidate set
- AND filters help enforce comparability

#### Scenario: Filter by execution state
- GIVEN runs in different states
- THEN filtering MUST allow excluding FAILED runs from metric comparisons
- BUT include them in overview listings with state

### Requirement: Baseline Documentation
The system MUST record which run/config is used as baseline and why, preserving baseline identity.

#### Scenario: Record baseline selection
- GIVEN baseline is selected
- THEN baseline run_id/config_id, timestamp of selection, and rationale MUST be recorded

#### Scenario: Baseline immutability in context
- GIVEN baseline is established for a comparison
- THEN the baseline reference MUST remain stable for that comparison
- AND changes to baseline selection MUST create a new comparison context with rationale

### Requirement: Comparison Output Structure
The system MUST produce structured comparison outputs including master table, baseline comparison table, and visibility into warnings.

#### Scenario: Produce baseline comparison table
- GIVEN baseline and candidates
- THEN table with run identity, state, key metrics, deltas vs baseline, comparability flags, warnings MUST be produced

#### Scenario: Include warnings and flags
- GIVEN non-comparable pairs exist
- THEN comparability warnings, controlled/multi-variable flags MUST appear in output

#### Scenario: No hidden ranking
- GIVEN multiple candidates
- THEN ordering MUST not be forced by a composite score
- AND any ordering applied MUST be explicit and justified by a single chosen metric if used