# Design: prebenchmark-infra

## Technical Approach

Package-first extraction. A new `prebenchmark/` package generalizes the pipeline duplicated across `Experimentos/*.py` (seed → split → scale → MLP → He init → full-batch train → test metrics) and adds the missing infrastructure: explicit parameter space, VALID/WARNING/INVALID pre-validation, run identity, hybrid persistence, comparator, repetition aggregation, manual decision log, and CSV exports. Legacy scripts stay **untouched** (AC 13 + protected uncommitted `EPOCHS=1000` change in `wine_quality.py`); reuse is by extraction, not import. Mini-batch `DataLoader` and a validation split are introduced deliberately — legacy prints were never persisted, so no comparability is lost (documented in metadata). Delivery is six chained PRs in requirements §15 order.

## Architecture Decisions

| Decision | Options | Choice & rationale |
|---|---|---|
| Two identifiers vs one | single `config_id`; `config_id` + `space_key` | **Two.** `config_id` (full config) satisfies "any differing field → different id"; `space_key` (4 space dims) governs 36-space membership so `class_weight` does not split the space. The spec's two id scenarios are otherwise in tension. |
| `config_id` derivation | random UUID; content hash | **Content hash.** `cfg-v1-<sha256(canonical JSON)[:12]>` — stable across runs/machines, enabling grouping without a registry; version prefix allows schema evolution. |
| Persistence store | SQLite; CSV-only; per-run dirs + CSV | **Per-run dirs + CSV.** `run_id` = directory name makes "never overwrite" structurally true; transparent, portable, pandas-native. SQLite deferred (documented). |
| Config storage | inline in every metric row; once + snapshot | **Once in `configs.jsonl` + full snapshot in `metadata.json`.** Snapshot guarantees reproducibility/immutability; log dedups. |
| Epoch write strategy | append per epoch; buffer + single write | **Buffer in memory, write once at finalize.** 1000 rows is trivial; guarantees epoch-history immutability even on failure (partial buffer still written). |
| `min_repetitions` default | 2; 3; 5 | **3**, configurable. Minimal n to report a sample std without a significance claim; conservative, explicit in outputs. |
| Run classification | auto engine; manual only | **Manual only** (binding). Auto-rule engine is a documented extension point, not built. |
| Test runner | none; pytest | **pytest** added to `requirements.txt` (AC 12; `strict_tdd: false`). |

## Module Layout & Public Entry Points

| Module | Responsibility | Public API |
|---|---|---|
| `config.py` | Typed, serializable config model + nested specs; defaults | `ExperimentConfig`, `DatasetSpec`, `ArchitectureSpec`, `SplitSpec`, `PreprocessingSpec`, `ConvergenceSpec`; `.to_dict()/.from_dict()` |
| `space.py` | Space definition, 36-combo generation, validation | `ParameterSpace`, `generate_configs(space)`, `validate_config(cfg, rules=None) -> ValidationResult` |
| `identity.py` | Stable/unique identifiers | `compute_config_id(cfg)`, `compute_space_key(cfg)`, `new_run_id()`, `new_experiment_id(...)` |
| `store.py` | Persistence; no-overwrite | `ResultsStore(root)`: `persist_run(...)`, `load_run(id)`, `query_runs(**filters)`, `append_decision(rec)`, `load_decisions(**filters)`, `register_config(cfg, validation)` |
| `data.py` | Dataset load (call scope, cached), splits, loaders | `load_wine_quality(cache_dir)`, `fetch_wine_quality_csv()`, `build_splits(cfg)`, `build_dataloaders(...)` |
| `model.py` | Architecture + weight init | `build_model(arch, activation)`, `initialize_weights(model, init, activation)` |
| `metrics.py` | Final + per-epoch metric computation | `compute_final_metrics(y_true, y_pred, loss)`, `epoch_train_step(...)`, `evaluate(...)` |
| `runner.py` | Reproducible run orchestration | `run_experiment(cfg, experiment_id, store) -> RunResult` |
| `compare.py` | Baseline, guards, deltas | `select_baseline(...)`, `check_comparability(a, b)`, `compare_to_baseline(...)` |
| `aggregate.py` | Repetition grouping + stats | `group_runs(...)`, `aggregate_group(..., min_repetitions=3)` |
| `decisions.py` | Append-only decision log | `record_decision(...)` |
| `export.py` | CSV exports | `export_all(store, experiment_id, out_dir)` |
| `cli.py` | Thin entry point (`python -m prebenchmark`) | generate / run / compare / aggregate / export |

`prebenchmark/__init__.py` re-exports the public API and performs **no** I/O on import (import-safe).

## Data Model

```
ExperimentConfig:
  run_type: "prebenchmark"|"benchmark"
  dataset: DatasetSpec(name="wine_quality", source="UCI winequality-white.csv",
                       version=<retrieved-id>, target="quality", n_features=11)
  architecture: ArchitectureSpec(name="wine_quality_mlp", input_dim=11,
                                 hidden_layers=[32,16], output_dim=7)
  activation: "relu"|"sigmoid"|"tanh"
  optimizer: "sgd"|"adam"
  lr: float
  init: "xavier"|"he"
  loss: "cross_entropy"
  batch_size: int = 32
  max_epochs: int = 500
  seed: int = 42
  split: SplitSpec(train=0.6, val=0.2, test=0.2, stratify=True)
  preprocessing: PreprocessingSpec(scaler="standard", class_weight=None|"balanced")
  convergence: ConvergenceSpec(enabled=False, metric=None, patience=None,
                               min_delta=None, mode="min")
  device: "auto"|"cpu"|"cuda"
```

- **Metadata (`metadata.json`)**: run_id, config_id, space_key, experiment_id, run_type, start/end timestamps, full config snapshot, dataset+version, seed, architecture, optimizer, lr, init, batch_size, max_epochs, epochs_executed, loss, split proportions+sizes, convergence cfg+trigger, preprocessing, device, hardware, library versions, execution state, warnings[], failure{reason, traceback}, final metrics (evaluation-set labelled), confusion matrix (nested lists), n_params, training_time.
- **`epochs.csv`**: `run_id, config_id, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time` (val = empty/NaN when absent).
- **Decision record (`decisions.jsonl`)**: `decision_id, entity{run_id|config_id|group_id}, decision_type, label, justification (required non-empty), actor, timestamp, evidence`.
- **Config record (`configs.jsonl`)**: `config_id, space_key, full params, preprocessing, convergence, validation_state, validation_rationale[], timestamp`.

## Data Flow

```
ParameterSpace ──generate_configs──→ [ExperimentConfig × 36]
       │                                     │
       └────────── validate_config ──────────┤ VALID/WARNING (executable) │ INVALID (blocked)
                                             ▼
   run_experiment ── identity.new_run_id ──► train loop ──► RunResult
       │  data.load_wine_quality (call scope, cached)                    │
       │  build_splits → build_dataloaders → build_model → init → optim   │
       └────────────────────────── store.persist_run ◄───────────────────┘
                                             │
        compare (baseline + guards) ─────────┤
        aggregate (group by config_id) ──────┤
        decisions (append-only) ─────────────┤
                                             ▼
                              export.export_all → results/exports/<experiment>/
```

## Persistence Layout

```
results/<experiment-id>/
├── experiment.json          # experiment metadata + baseline selection + rationale
├── configs.jsonl            # append-only config defs + validation states
├── decisions.jsonl          # append-only decision log
└── <run_id>/
    ├── metadata.json        # full metadata + config snapshot + final metrics
    └── epochs.csv           # immutable per-epoch history
results/exports/<experiment-id>/{master,baseline_comparison,aggregates,epochs,configs,decisions,warnings_summary}_<UTCts>.csv
.cache/prebenchmark/winequality-white.csv   # dataset cache (gitignored)
```

**No-overwrite guarantee**: `new_run_id() = <UTC YYYYMMDDTHHMMSSZ>-<6 hex>`; the run directory is created with `os.mkdir` (atomic, fails if present) and the id is regenerated on collision. `metadata.json`/`epochs.csv` are written once and never modified. `results/` and `.cache/` are added to `.gitignore` (binding: never committed).

## Training Runner Design

Generalizes the legacy pipeline, changing only what configuration requires:

- **Model**: `nn.Sequential` built from `ArchitectureSpec`; hidden layers inject the configured activation, output layer is linear (matches legacy `Linear→act→Linear→act→Linear`).
- **Weight init**: biases zeroed. He → `kaiming_normal_` with `nonlinearity` mapped per activation (`relu→relu`, `tanh→tanh`, `sigmoid→sigmoid`); Xavier → `xavier_normal_`. Applied to every `nn.Linear`.
- **Optimizer/loss**: SGD or Adam at `cfg.lr`; `CrossEntropyLoss(weight=class_weights)` when `preprocessing.class_weight=="balanced"` (weights computed from train frequencies of the *training* split, indexed by class id so a class absent from train cannot misalign the vector), else `None`. The weighted criterion applies to the **training objective only**: validation and test evaluation always use the unweighted cross-entropy, so `val_loss` (the early-stopping signal) and the final `loss` measure the same quantity in every run and their differences stay comparable across a balanced and an unbalanced arm.
- **Data/batching**: `TensorDataset` + `DataLoader(batch_size, shuffle=True, generator=<seeded>)`; epoch loss = sample-weighted mean over batches; epoch accuracy = correct / total.
- **Per-epoch capture**: `train_loss`, `train_acc`, plus `val_loss`/`val_acc` when a val set exists (else N/A) and `epoch_time`; buffered and persisted once at finalize.
- **NaN/Inf detection**: per batch/epoch `torch.isnan/isinf` (per-batch train+val, weights each epoch); first offending epoch/batch/source recorded. Non-finite detection **aborts** the run as `FAILED` and it is still persisted (confirmed decision).
- **Epoch count**: `epochs_executed = len(epoch_records)`.
- **Convergence hook**: after each epoch, if `convergence.enabled`, apply the declared criterion; stop and record trigger (reason, metric, patience, min_delta, mode). Unspecified fields resolve to the recorded defaults `metric=val_loss, patience=20, min_delta=1e-4, mode=min` (see Open Questions — resolved). Default OFF → `max_epochs`.
- **Device**: `auto` → CUDA if available else CPU; model/tensors moved; actual device recorded.
- **Determinism**: `np.random.seed(seed)` + `torch.manual_seed(seed)` (+ `cuda.manual_seed_all`); DataLoader generator seeded; cuDNN nondeterminism noted in metadata.
- **Failure handling**: `try/except` around the loop; on exception record `FAILED`, traceback, and the partial epoch buffer, then finalize via the store (unique run_id; no overwrite).

**Baseline** (recorded, not silent): ReLU + Adam + lr 0.001 + He + CrossEntropyLoss + existing `wine_quality` architecture — the only configuration ever run in this repo.

## Comparator Design

- **Baseline selection**: default = config matching the baseline constant; explicit override allowed. run_id/config_id + timestamp + rationale recorded in `experiment.json` and a decision record.
- **Comparability guards**: compare dataset, split proportions, architecture, max_epochs, convergence, run_type, and evaluation set. Any mismatch → `non_comparable` flag + reason; metric deltas marked N/A (never emitted as if valid).
- **Controlled vs multi-variable**: diff the open dims {activation, optimizer, lr, init} plus `class_weight`; exactly one differs → `controlled` (variable of interest reported); >1 → `multi_variable`. Flag printed in output. `class_weight` does not split the 36-combination space, but it does change the training protocol, so two runs differing only in the balance are a controlled comparison — never "identical".
- **Deltas**: absolute Δ for accuracy, balanced accuracy, precision macro, recall macro, F1 macro, loss, training_time — same evaluation set only. Relative Δ only when the denominator is interpretable (non-zero, bounded); basis stated.
- **No composite score**: full metric profile + trade-offs preserved; no forced ranking.
- **Failed runs**: listed with state + reason; contribute no performance deltas.
- **Output**: comparison table `run_id, config_id, state, eval_set, metrics, Δ vs baseline, comparability flag, controlled flag, warnings`.

## Repetition Aggregation

- **Grouping**: by `config_id` (protocol settings are inside `config_id`; differing seeds share it). Group metadata: `config_id`, member run_ids + seeds, config snapshot, `created_at`.
- **Statistics** per metric (non-FAILED runs only): `n_valid, mean, std (ddof=1), min, max, range`; separately report `n_total, n_failed, n_unstable` and their proportions.
- **Insufficient-repetitions flag**: `n_valid < min_repetitions` (default **3**, configurable, recorded) → statistics flagged "insufficient repetitions"; values still shown but never presented as reliable.
- **Failed runs**: excluded from stats, never silently dropped — always disclosed in the summary.

## Validation Rules (VALID / WARNING / INVALID)

Rules are a configurable `ValidationRule` list with documented defaults:

- **INVALID** (real technical/methodological constraints only; execution blocked): unknown activation/optimizer/init/loss name; architecture dims incompatible with dataset (input/output mismatch); split proportions not summing to 1; `batch_size<1`; `max_epochs<1`; `lr<=0`.
- **WARNING** (executable, risk recorded): `lr=0.1` with Adam (instability) or SGD (divergence risk); Sigmoid/Tanh + He init and ReLU + Xavier (non-recommended pairing); NaN-prone combinations; `class_weight="balanced"` on already-balanced data.
- **No convention-based rejection**: non-recommended activation/init pairings are WARNING, never INVALID. Default rules leave **all 36 combos executable** (none INVALID), asserted by a test.

## Test Strategy

| Layer | What | Approach |
|---|---|---|
| Unit | config round-trip/defaults; space = 36 & extensibility; identity stability/sensitivity + `space_key` ignoring `class_weight`; validation states (36 non-INVALID); store no-overwrite/failed/epoch-immutability/query; compare deltas & guards; aggregate stats & flag; decisions append-only; exports columns/rows | pytest, pure functions, no training |
| Integration | reduced run (max_epochs=3) persists `metadata.json` + 3-row `epochs.csv`, state `COMPLETED`; NaN run marked & persisted | monkeypatch `fetch_wine_quality_csv` with synthetic wine_quality-shaped DataFrame (offline-safe); force CPU |
| E2E | full 36-run prebenchmark (excluded from CI); one committed reduced smoke marker test | documented manual command |

**Mock**: UCI download (fixture DataFrame), device detection (force CPU), optional plotting (skip if matplotlib missing). **Coverage goal**: pure modules ≥ 90%, overall ≥ 70% (target, not a gate — `coverage_threshold: 0`).

## File Changes

| File | Action | Description |
|---|---|---|
| `prebenchmark/*.py` (12 modules above) | Create | New package |
| `tests/*.py`, `tests/conftest.py` | Create | pytest suite + offline fixtures |
| `requirements.txt` | Modify | Add pinned `pytest` (+ `iniconfig`, `pluggy`); optional `pytest-cov` |
| `.gitignore` | Modify | Add `results/`, `.cache/` |
| `README.md` | Modify | Usage: define/run/compare/aggregate/export + test command |
| `Experimentos/*` | — | **Untouched** (protected) |

## PR Slicing (§15 order, ≤400 changed lines each, feature-branch chain)

| PR | Delivers | Verification |
|---|---|---|
| 1 | `config.py`, `space.py`, pytest bootstrap, `.gitignore`, tests | 36 combos; none INVALID; config round-trip; validation states |
| 2 | `identity.py`, `store.py`, decision scaffolding, tests | id stability/sensitivity; no-overwrite; failed-run + immutability tests |
| 3 | `data.py`, `model.py`, `metrics.py`, `runner.py`, tests | offline reduced smoke persists metadata+epochs; NaN detection |
| 4 | `compare.py`, tests | delta correctness; comparability flags; controlled/multi-variable |
| 5 | `aggregate.py`, `decisions.py`, tests | stats; insufficient flag; append-only decision log |
| 6 | `export.py`, `cli.py`, docs, reduced smoke | exports with expected columns; end-to-end reduced run; suite green |

## Migration / Rollout

No migration required — greenfield package; new directories only. `Experimentos/` untouched. Rollout: merge PR chain in order; run reduced smoke after PR3, then PR6; run the full 36-config prebenchmark only after verify. Rollback: revert the chain in reverse; delete `prebenchmark/`, `tests/`, `results/`; restore `requirements.txt`/`.gitignore`.

## Open Questions

- [x] **Resolved** — He/Xavier variants: the baseline is ReLU + He/Kaiming (`kaiming_normal_`, biases zeroed), confirmed as the documented default. Xavier (`xavier_normal_`) remains available for `tanh`/`sigmoid` activations. See `initialize_weights()` in `prebenchmark/model.py`.
- [x] **Resolved** — enabling convergence: defaults are now declared and applied when a field is unspecified: `metric=val_loss, patience=20, min_delta=1e-4, mode=min`, plus `restore_best_weights=True` (the best-validation epoch weights are restored at the end, mirroring Keras semantics). The criterion stays OFF by default; the resolved criterion plus the trigger reason are recorded in metadata (see `resolve_convergence()` in `prebenchmark/config.py` and `run_experiment()` in `prebenchmark/runner.py`).
- [x] **Resolved** — multi-seed protocol for stage 1 (exploratory): each configuration is run with the fixed seed set `(42, 123, 7, 2025, 999)` and compared by mean ± sample std over `val_loss` and `val_accuracy`. Later stages remain single-run until re-evaluated. The fixed set is defined in `prebenchmark/seeds.py` (`EXPLORATORY_SEEDS`).

