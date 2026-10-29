# Tasks: prebenchmark-infra

> openspec | ask-always | 6 chained slices (design §15). Legacy `Experimentos/*` read-only (AC13); `results/`,`.cache/` gitignored; `strict_tdd=false` (pytest added in PR1).

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~2,170 (PR1 380 / PR2 360 / PR3 420 / PR4 300 / PR5 330 / PR6 380) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR1→PR2→PR3→PR4→PR5→PR6 |
| Delivery strategy | ask-always |
| Chain strategy | stacked-to-main (orchestrator-resolved) |

```text
Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High
```

### Suggested Work Units

| Unit | Goal | PR | AC | Base |
|------|------|----|----|------|
| 1 | Config model + 36-space + validation + pytest | PR1 | 1,2,12 | tracker |
| 2 | Identity + hybrid persistence + failures | PR2 | 3,4,5,6,12 | PR1 |
| 3 | Data/model/metrics/runner + reduced smoke | PR3 | 4,5,12,13 | PR2 |
| 4 | Baseline comparator + guards | PR4 | 7,8,10,12 | PR3 |
| 5 | Aggregation + decision log | PR5 | 9,11,12 | PR4 |
| 6 | CSV exports + CLI + docs + E2E smoke | PR6 | 10,12,13 | PR5 |

## Phase 1 — Config Space & Test Bootstrap (PR1)

- [x] 1.1 `requirements.txt`: pin `pytest`,`iniconfig`,`pluggy`; `.gitignore`: add `results/`,`.cache/`. (AC12)
- [x] 1.2 `pytest.ini` + `tests/conftest.py`: offline wine_quality fixture, force CPU. (AC12)
- [x] 1.3 `prebenchmark/__init__.py`: import-safe re-export, no I/O. (training-runner Import-safe)
- [x] 1.4 `prebenchmark/config.py`: `ExperimentConfig`,`DatasetSpec`,`ArchitectureSpec`,`SplitSpec`,`PreprocessingSpec`,`ConvergenceSpec` + `.to_dict()/.from_dict()`; defaults 60/20/20, bs=32, epochs=1000, seed=42, convergence OFF. (config-space Config Model/Split/Convergence; AC1)
- [x] 1.5 `prebenchmark/space.py`: `ParameterSpace`,`generate_configs()`=36,`validate_config()`,`ValidationResult`,`ValidationRule`; VALID/WARNING/INVALID + rationale, 36 non-INVALID. (config-space Generation/Validation; AC2)
- [x] 1.6 `tests/test_config.py`,`test_space.py`: round-trip/defaults, 36 unique, extensibility, validation states, class_weight recorded but excluded from space_key. (AC1,2,12)

## Phase 2 — Identity & Persistence (PR2)

- [x] 2.1 `prebenchmark/identity.py`: `compute_config_id`(sha256 canonical JSON,`cfg-v1-`),`compute_space_key`(4 dims),`new_run_id`,`new_experiment_id`. (run-persistence Run Identity; AC3)
- [x] 2.2 `prebenchmark/store.py`: `ResultsStore(root)`: `persist_run`,`load_run`,`query_runs`,`register_config`,`append_decision`,`load_decisions`; per-run dir via `os.mkdir`,`configs.jsonl`,`decisions.jsonl`. (run-persistence No Overwrite/Storage/Query; AC3,4,5,6)
- [x] 2.3 `tests/test_identity.py`,`test_store.py`: stability/sensitivity, space_key ignores class_weight, no-overwrite, failed+partial persistence, epoch immutability, query filters. (AC3,5,6,12)

## Phase 3 — Training Runner (PR3)

- [x] 3.1 `prebenchmark/data.py`: `load_wine_quality(cache_dir)`(call-scope,cached),`fetch_wine_quality_csv`,`build_splits`(60/20/20,stratify),`build_dataloaders`(bs=32,seeded); never import `Experimentos/`. (training-runner Dataset/Split/DataLoader; AC13)
- [x] 3.2 `prebenchmark/model.py`: `build_model(arch,activation)`,`initialize_weights(model,init,activation)` (He→kaiming, Xavier→xavier_normal; biases 0). (training-runner Model Init; AC13)
- [x] 3.3 `prebenchmark/metrics.py`: `compute_final_metrics`,`epoch_train_step`,`evaluate` — accuracy, balanced acc, precision/recall/F1 macro, confusion, loss. (training-runner Final Eval; AC4)
- [x] 3.4 `prebenchmark/runner.py`: `run_experiment(cfg,experiment_id,store)->RunResult`; seed np+torch, optimizer/loss, per-epoch buffer, NaN/Inf detect, convergence hook, device, failure capture. (training-runner Reproducible/Per-Epoch/Convergence/Numerical/Failure; AC4,5,13)
- [x] 3.5 `tests/test_data.py`,`test_model.py`,`test_metrics.py`,`test_runner.py`: offline max_epochs=3 smoke persists `metadata.json`+3-row `epochs.csv` (`COMPLETED`); NaN run flagged+persisted. (AC4,5,12,13)

## Phase 4 — Comparator (PR4)

- [x] 4.1 `prebenchmark/compare.py`: `select_baseline`,`check_comparability`,`compare_to_baseline`; abs/rel deltas, guards (dataset/split/arch/epochs/convergence), controlled vs multi-variable, no composite. (results-comparison; AC7,8,10)
- [x] 4.2 `tests/test_compare.py`: delta correctness, guard flags, controlled/multi-variable, failed exclusion + reason. (AC7,8,10,12)

## Phase 5 — Aggregation & Decisions (PR5)

- [x] 5.1 `prebenchmark/aggregate.py`: `group_runs`(config_id),`aggregate_group(min_repetitions=3)`; n/mean/std(ddof=1)/min/max/range, insufficient-repetitions flag, failed/stability proportions. (repetition-aggregation; AC9)
- [x] 5.2 `prebenchmark/decisions.py`: `record_decision`; append-only `decisions.jsonl`, required justification, actor, timestamp, entity link. (run-classification-decisions; AC11)
- [x] 5.3 `tests/test_aggregate.py`,`test_decisions.py`: stats+insufficient flag, multi-seed grouping, failed counts, append-only, justification required, state vs label independence. (AC9,11,12)

## Phase 6 — Exports, CLI & Docs (PR6)

- [x] 6.1 `prebenchmark/export.py`: `export_all(store,experiment_id,out_dir)` → master/baseline_comparison/aggregates/epochs/decisions/configs/warnings_summary CSVs; N/A consistency; optional plots non-blocking. (results-export; AC10)
- [x] 6.2 `prebenchmark/cli.py`: `python -m prebenchmark {generate,run,compare,aggregate,export}`. (AC13)
- [x] 6.3 `tests/test_export.py`,`test_cli.py` + committed reduced smoke marker: expected columns/rows, no field loss, E2E reduced run. (AC12,13)
- [x] 6.4 `README.md`: usage, persistence location, test command, pending decisions. (AC13)

AC map: 1,2→PR1; 3–6→PR2–3; 7,8,10→PR4; 9,11→PR5; 12→all; 13→PR3+PR6.
