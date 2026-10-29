# neural-network-benchmark

Experimental registration, execution, comparison and export infrastructure for the
neural-network benchmark (pre-benchmark phase).

The pipeline is **register → run → compare → aggregate → export**, all driven by a
single package and persisted as append-only artifacts so every claim is traceable
back to the exact configuration and data snapshot that produced it.

## Requirements

```
pip install -r requirements.txt
```

## CLI usage

Every subcommand shares `--experiment-id` (required) and `--root` (results store
root, default `results`).

```bash
# 1. Register candidate configurations (optionally limit to the first N)
python -m prebenchmark generate --experiment-id exp-001 --limit 36

# 2. Execute configurations (limit = how many to run; override epochs/seed)
python -m prebenchmark run --experiment-id exp-001 --limit 36 --max-epochs 200

# 2b. Exploratory stage: run each configuration once per fixed seed set (5 seeds)
python -m prebenchmark run --experiment-id exp-001 --limit 36 --max-epochs 200 --multi-seed

# 3. Compare every run against a baseline (explicit run/config id, or documented default)
python -m prebenchmark compare --experiment-id exp-001 --documented-default

# 4. Aggregate repeated configurations (multi-seed variability)
python -m prebenchmark aggregate --experiment-id exp-001 --grouping config

# 5. Export all result tables as timestamped CSVs
python -m prebenchmark export --experiment-id exp-001
```

## Persistence location

Results are written under the store root (`results/` by default):

```
results/
  <experiment-id>/
    <run-id>/
      metadata.json      # run record: config snapshot, dataset version, env, state
      epochs.csv         # per-epoch train/val history
    configs.jsonl        # append-only configuration registry (one per config_id)
    decisions.jsonl      # append-only decision log (justification required)
  exports/               # timestamped CSV tables from `export`
```

## What each run records

- **Dataset version** — a `sha256-<digest>` fingerprint of the dataset bytes (or the
  in-memory DataFrame's canonical serialization), so two runs can be proven to have
  used the same data.
- **Environment** — OS/CPU/machine and CUDA availability, plus library versions
  (Python, torch, numpy, pandas, scikit-learn). Captured best-effort; never blocks a run.
- **Execution state** — the canonical `execution_state` field is written for every
  outcome: `COMPLETED`, `COMPLETED_WITH_WARNINGS` or `FAILED`. The legacy `state`
  spelling is accepted as a query alias.
- **Convergence** — off by default. When enabled, unspecified fields are filled with
  the documented defaults (metric `val_loss`, patience `20`, min_delta `1e-4`,
  mode `min`, `restore_best_weights=True`) and the resolved criterion plus the trigger
  reason are recorded.
- **Validation metrics** — every run records `validation_metrics` for the best epoch
  (lowest `val_loss`): `{evaluation_set, epoch, loss, accuracy}`. These are the
  metrics used to compare configurations across seeds in the exploratory stage, so the
  test set stays reserved for final evaluation.
- **Numerical robustness** — non-finite (NaN/Inf) losses are detected per batch during
  train and validation, and weights are checked each epoch. A non-finite value aborts
  the run as `FAILED`, recording the first failing epoch, batch and source.

## Tests

```bash
python -m pytest -q
```

The suite runs fully offline (in-memory datasets; no network required).

## Resolved decisions

From `openspec/changes/prebenchmark-infra/design.md` (Open Questions) — these were
previously left to the researchers and are now confirmed:

- **Initialization**: baseline is ReLU + He/Kaiming (`kaiming_normal_`, biases zeroed);
  Xavier (`xavier_normal_`) stays available for `tanh`/`sigmoid`.
- **Convergence defaults** (applied only when the criterion is enabled):
  `metric=val_loss`, `patience=20`, `min_delta=1e-4`, `mode=min`, and
  `restore_best_weights=True`. The criterion stays OFF by default.
- **Default `max_epochs=500`**: lowered from 1000 after an exploratory timing sweep
  showed the winning configurations converge well before 500 epochs (best epoch in the
  16-303 range) and that the only slow region (`lr=0.001` + `sgd`) was truncated at the
  cap anyway and still reached worse `val_loss` than fast-converging configs. The cap is
  a termination budget, not a convergence target; early stopping handles the rest.
- **Multi-seed protocol (stage 1, exploratory)**: each configuration is run with the
  fixed seed set `(42, 123, 7, 2025, 999)` and compared by mean ± sample std over
  `val_loss` and `val_accuracy`. Later stages remain single-run until re-evaluated.
  The fixed set lives in `prebenchmark/seeds.py` (`EXPLORATORY_SEEDS`).
