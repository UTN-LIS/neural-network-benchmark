# Delta for run-classification-decisions

## ADDED Requirements

### Requirement: Execution State Classification
The system MUST classify each run into an execution state: `COMPLETED`, `COMPLETED_WITH_WARNINGS`, `FAILED`, or `INCOMPLETE` as appropriate.

#### Scenario: Mark successful completion
- GIVEN a run completes without errors or warnings
- THEN execution state MUST be `COMPLETED`

#### Scenario: Mark completed with warnings
- GIVEN a run completes but encounters warnings (e.g., numerical issues, non-critical conditions)
- THEN execution state MUST be `COMPLETED_WITH_WARNINGS`
- AND all warnings MUST be recorded

#### Scenario: Mark failed
- GIVEN a run terminates due to error/exception
- THEN execution state MUST be `FAILED`
- AND failure reason MUST be recorded

#### Scenario: Track warnings
- GIVEN warnings occur
- THEN each warning MUST include message, category, and context (epoch if applicable)
- AND warnings MUST be persisted with the run

### Requirement: Exploratory Evaluation Classification
The system MUST support exploratory evaluation labels independent from execution state: `VIABLE`, `INTERESTING`, `UNSTABLE`, `NON_VIABLE`, `PENDING_REVIEW`.

#### Scenario: Assign exploratory label
- GIVEN a completed run
- THEN it MAY be assigned an exploratory label
- AND the label MUST be independent of technical execution state

#### Scenario: Allow pending review
- GIVEN results are inconclusive
- THEN label MAY be `PENDING_REVIEW`

#### Scenario: Distinguish technical vs exploratory
- GIVEN a run is `COMPLETED` but shows instability
- THEN exploratory label MAY be `UNSTABLE`
- AND execution state remains `COMPLETED`

### Requirement: Manual Decision Recording
The system MUST record manual decisions about runs with full justification, actor, and timestamp.

#### Scenario: Record manual decision
- GIVEN a user/agent makes a decision (keep, exclude, repeat, investigate)
- THEN the decision MUST be recorded with justification, who/what process made it, and timestamp
- AND it MUST be linked to the run_id

#### Scenario: Require justification
- GIVEN any decision is recorded
- THEN a non-empty justification MUST be provided
- AND the justification MUST explain the reasoning

#### Scenario: Track actor
- GIVEN manual decision
- THEN actor identification (human or system process name) MUST be recorded when available
- AND the capability of the project determines the level of detail recorded

### Requirement: Decision Log Structure
The system MUST maintain an append-only decision log with entries for each decision made about runs/configurations/groups.

#### Scenario: Append-only log
- GIVEN decisions are made over time
- THEN new decisions MUST be appended
- AND existing decision entries MUST NOT be modified or deleted

#### Scenario: Link to entities
- GIVEN a decision
- THEN it MUST reference the relevant entity (run_id and/or config_id and/or group_id)
- AND include decision type, label, justification, actor, timestamp

#### Scenario: Support decision history
- GIVEN a run has multiple decisions over time
- THEN all decisions MUST be preserved in order
- AND the full history MUST be retrievable

### Requirement: No Automatic Discard Based on Single Metric
The system MUST NOT automatically discard configurations/runs solely because a single metric is inferior.

#### Scenario: Preserve on inferior single metric
- GIVEN a run has lower accuracy but better recall/F1 or different trade-off
- THEN it MUST NOT be automatically marked as NON_VIABLE
- AND trade-offs MUST be considered in evaluation

#### Scenario: Require justification for exclusion
- GIVEN a run is marked as excluded/non-viable
- THEN a justification MUST be recorded explaining why (beyond "metric X lower")

#### Scenario: Preserve unexpected results
- GIVEN results contradict conventional initialization/optimization expectations
- THEN the run MUST remain preserved for review
- AND the decision log MAY record the observation

### Requirement: Pre-Validation Decision Recording
The system MUST record validation outcomes for candidate configurations (VALID/WARNING/INVALID) with rationales.

#### Scenario: Record validation result
- GIVEN a configuration is validated
- THEN validation state, rationale for WARNING/INVALID, and timestamp MUST be recorded
- AND linked to config_id

#### Scenario: WARNING executions tracked
- GIVEN a WARNING configuration is executed
- THEN the warning rationale MUST be linked to the resulting run
- AND remain visible in run record

### Requirement: Stability Assessment Support
The system MUST support recording stability observations based on metrics and variability.

#### Scenario: Mark unstable based on evidence
- GIVEN a run shows signs of instability (divergence, oscillation indicators, NaN)
- THEN it MAY be labeled `UNSTABLE`
- AND evidence supporting the label MUST be recorded

#### Scenario: Distinguish from failure
- GIVEN instability without hard failure
- THEN execution state MAY remain `COMPLETED_WITH_WARNINGS`
- AND exploratory label `UNSTABLE`

### Requirement: Decision Query and Export
The system MUST support querying and exporting decision log entries.

#### Scenario: Query decisions by run
- GIVEN a run_id
- THEN all decisions related to that run MUST be retrievable

#### Scenario: Export decision log
- GIVEN decision entries exist
- THEN decision log MUST be exportable to CSV/structured format with all fields

#### Scenario: Filter decisions
- GIVEN decision log
- THEN filtering by decision type, label, actor, date range MUST be supported

### Requirement: Configurable Classification Rules with Defaults
The system MUST allow configurable rules for exploratory classification, with documented defaults and no hidden arbitrary cut-offs.

#### Scenario: Document default rules
- GIVEN default classification rules are applied
- THEN they MUST be documented with justification
- AND defaults MUST be explicitly stated

#### Scenario: Make rules configurable
- GIVEN classification logic uses thresholds
- THEN thresholds MUST be configurable
- AND any threshold value used MUST be visible/documented

#### Scenario: No hidden cut-offs
- GIVEN classification is performed
- THEN the rules applied MUST be recorded or referenced
- AND decisions based on rules MUST still allow manual override with justification