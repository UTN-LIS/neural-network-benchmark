# Proposal: prebenchmark-infra

## Intent

Narrow 36 combos before the main benchmark; scripts hardcode one config, persist nothing, lack baseline/tests.

## Scope

### In Scope
- `prebenchmark/` package: space, store, runner, comparison, aggregation, exports.
- `wine_quality` only; 36 combos; 1000 epochs, DataLoader bs=32, configurable split, SEED=42.
- pytest; chained PRs.

### Out of Scope
- Any `Experimentos/` edit (esp. `wine_quality.py`) — reuse by extraction, never import.
- Automatic classification (manual, justified); balancing in space; other datasets; multi-seed runs later; web UI; SQLite.

## Capabilities

### New Capabilities
- `experiment-config-space`: config model, 36-combo generation, VALID/WARNING/INVALID validation.
- `run-persistence`: config_id/run_id, metadata, epoch metrics, failure records, no overwrite.
- `training-runner`: reproducible run, DataLoader, val split, per-epoch/final metrics, env/device.
- `results-comparison`: baseline deltas, controlled comparisons, comparability guards, no composite score.
- `repetition-aggregation`: grouping, statistics, insufficient-repetition flags.
- `run-classification-decisions`: run states + manual evaluation, recorded justification.
- `results-export`: master/baseline/aggregate tables, CSV exports.

### Modified Capabilities
None — specs/ empty.

## Approach

Package-first extraction; §15 module order: `space`, `store`, `runner`, `compare`, `aggregate`, `export`.

Hybrid persistence: per-run dir (`metadata.json` + `epochs.csv`) keyed by `config_id` + CSV aggregates; overwrite impossible; SQLite deferred. `wine_quality` loaded at call scope with cache — never via legacy import.

## Affected Areas

- New: `prebenchmark/`, `tests/`
- Modified: `requirements.txt` (+pytest), `.gitignore`
- Untouched: `Experimentos/*`

## Delivery Slicing (§15 order)

| PR | Contents |
|----|----------|
| 1 | config generation + validation + pytest |
| 2 | identity + persistence + failures |
| 3 | runner, split, metrics, loader |
| 4 | comparator: baseline + guards |
| 5 | aggregation + decision log |
| 6 | exports + docs + smoke run |

## Test Strategy

pytest on pure modules; reduced-epoch smoke before the full run; specs require tests (AC 12).

## Acceptance Criteria Mapping

1–2→PR1; 3–6→PR2–3; 7–8→PR4; 9–11→PR5; 12→every PR; 13→PR3+PR6.

## Assumptions (correctable)

1. Split 60/20/20 (train/val/test), configurable.
2. Class balancing (`class_weight`) = optional config field, outside the 36-space.
3. Smoke run first; full 36-run after verify.
4. Baseline = ReLU + Adam + lr 0.001 + He init + CrossEntropyLoss + existing wine_quality architecture.

## Pending Decisions (not invented)

Split proportions if 60/20/20 rejected; balancing strategy (`class_weight` vs resampling); convergence criteria (config-declared, default off); `results/` gitignore vs commit; UCI offline story.

## Risks

| Risk | Mitigation |
|------|------------|
| `wine_quality.py` edited | No-touch rule; diff/PR |
| 400-line budget | Chained slicing |
| UCI network failure | Call-scope fetch + cache |
| Cross-protocol comparison | Comparability guards |

## Rollback Plan

Revert PR chain in reverse; delete new dirs; restore `requirements.txt`/`.gitignore`.

## Success Criteria

- [ ] 13 ACs verified; pytest green
- [ ] Runs persisted, compared, exported; zero `Experimentos/` diff
