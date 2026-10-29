# Delta for repetition-aggregation

## ADDED Requirements

### Requirement: Repetition Grouping
The system MUST group runs by configuration identity (same config_id) and protocol to form repetition groups.

#### Scenario: Group same configuration
- GIVEN multiple runs share identical configuration parameters
- THEN they MUST be assigned to the same repetition group
- AND the group MUST be keyed by `config_id`

#### Scenario: Distinguish different protocols
- GIVEN same parameters but different protocol settings affecting comparability
- THEN they MUST be treated as different groups or clearly separated
- AND grouping logic MUST be explicit

### Requirement: Aggregate Statistics Per Group
The system MUST compute aggregate statistics for each repetition group including count, mean, standard deviation, min, max for comparable metrics.

#### Scenario: Compute basic statistics
- GIVEN a group with N valid runs
- THEN for each metric (accuracy, balanced accuracy, precision macro, recall macro, F1 macro, loss, training time) the system MUST compute count of valid runs, mean, standard deviation, minimum, and maximum

#### Scenario: Only include valid runs
- GIVEN a group contains FAILED runs
- THEN FAILED runs MUST NOT be included in mean/std/min/max calculations
- AND the count of failed runs MUST be reported separately

#### Scenario: Report run counts explicitly
- GIVEN a group
- THEN the system MUST report total runs, valid runs, failed runs, and unstable runs (if classified)

### Requirement: Insufficient Repetitions Handling
The system MUST detect when repetition count is insufficient and explicitly flag aggregated statistics as unreliable.

#### Scenario: Flag insufficient repetitions
- GIVEN a group has fewer than the configurable minimum number of valid repetitions
- THEN aggregated statistics MUST be flagged as "insufficient repetitions"
- AND the minimum threshold used MUST be recorded

#### Scenario: Configurable minimum
- GIVEN no minimum specified
- THEN a default minimum MUST be applied and documented
- AND the threshold MUST be configurable (not hardcoded as an invisible rule)
- AND the insufficient-repetition flag MUST always be explicit in outputs

#### Scenario: Do not fabricate confidence
- GIVEN insufficient valid repetitions
- THEN the system MUST NOT present statistics as if they were statistically reliable
- AND the flag MUST be clearly visible in aggregate tables

### Requirement: Failed Run Treatment in Aggregates
The system MUST explicitly account for failed and unstable runs in group summaries without silently dropping them.

#### Scenario: Track failure rates
- GIVEN a group
- THEN the system MUST report number and proportion of failed runs
- AND number/proportion of unstable runs if applicable

#### Scenario: Preserve visibility
- GIVEN failures exist
- THEN they MUST appear in group summary
- AND statistics MUST clearly indicate which runs contributed to aggregates

### Requirement: Variability Reporting
The system MUST report variability between repetitions to support stability assessment.

#### Scenario: Report standard deviation
- GIVEN valid repetitions exist (meeting minimum or not)
- THEN standard deviation MUST be reported for key metrics
- AND when insufficient repetitions, the std value MUST still be reported but flagged

#### Scenario: Report range
- GIVEN min and max
- THEN the range (max-min) MUST be available to indicate spread

### Requirement: Group-Level Metadata
The system MUST record group identity, member run_ids, and configuration snapshot for each group.

#### Scenario: Record group membership
- GIVEN a repetition group
- THEN all member run_ids MUST be listed
- AND the shared `config_id` and configuration snapshot MUST be stored

#### Scenario: Track group creation time
- GIVEN a group is formed
- THEN timestamp/context of aggregation MUST be recorded

### Requirement: Support for Multiple Seeds
The system MUST support grouping runs with different seeds under the same configuration, enabling variability analysis.

#### Scenario: Group across seeds
- GIVEN same config executed with different `random_state` values
- THEN they MUST belong to the same repetition group
- AND seed values of each member MUST be preserved in member list

#### Scenario: Distinguish seed in member details
- GIVEN group members
- THEN each member's seed and run_id MUST be visible

### Requirement: Aggregate Export
The system MUST export aggregated results per repetition group in structured format (CSV).

#### Scenario: Export aggregate table
- GIVEN repetition groups exist
- THEN aggregate table with group key, config_id, member counts (total/valid/failed), mean/std/min/max for metrics, and insufficient-repetitions flags MUST be exportable to CSV

#### Scenario: Include configuration summary
- GIVEN export
- THEN key configuration dimensions (activation, optimizer, lr, init, dataset, batch size) MUST be included for each group for readability

### Requirement: Per-Group Comparison Readiness
The system MUST make aggregated group statistics available for comparison while preserving per-run detail when needed.

#### Scenario: Compare groups
- GIVEN multiple repetition groups
- THEN group-level statistics MAY be compared
- AND comparability guards (§8.2) MUST still apply at group level

#### Scenario: Drill down to runs
- GIVEN a group
- THEN the system MUST allow retrieving member run details without losing group context