# neural-network-benchmark

Experimental registration, execution, comparison and export infrastructure for the
neural-network benchmark (pre-benchmark phase).

The pipeline is **register → run → compare → aggregate → export → analyze**, all driven by a
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

# 6. Build a self-contained statistical report from an export snapshot
python -m prebenchmark analyze --experiment-id exp-001
```

`analyze` reads the latest `master_*.csv`, `configs_*.csv` and `epochs_*.csv` in
`results/exports/<experiment-id>/` (override with `--export-dir` or `--exports`)
and writes a single offline HTML file to
`results/reports/<experiment-id>/analysis_<timestamp>.html`. It never reads the
internal store, so the report is reproducible from the exported snapshot alone.

## Datasets

`generate` and `run` accept `--dataset NAME` (default `wine_quality`). The
registry lives in `prebenchmark/data.py`; `input_dim` and `output_dim` are
derived from the loaded dataset, never hardcoded. Every dataset is cached as a
canonical CSV under `.cache/prebenchmark/` so `dataset_version` remains a stable
`sha256-<digest>` fingerprint.

| Name | Source | Samples | Features | Classes | Max/min class | App menu |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| `wine_quality` | UCI winequality-white | 4,898 | 11 | 7 | 439.6 | yes |
| `satimage` | OpenML satimage v1 | 6,430 | 36 | 6 | 2.45 | yes |
| `vehicle` | OpenML vehicle v1 | 846 | 18 | 4 | 1.10 | yes |
| `waveform_5000` | OpenML waveform-5000 v1 | 5,000 | 40 | 3 | 1.02 | no |
| `segment` | OpenML segment v1 | 2,310 | 19 | 7 | 1.00 | no |
| `digits` | scikit-learn `load_digits` | 1,797 | 64 | 10 | 1.05 | no |

The **app menu** is the curated subset in `prebenchmark.config.DATASET_MENU`
(`wine_quality`, `satimage`, `vehicle`) and is what `/api/options` exposes to the
UI. The remaining datasets stay registered for the CLI and for tests, but they
are not offered as first-class comparison options: `segment` and `digits` are
saturated (simple baselines reach the MLP ceiling) and `waveform_5000` is
noise-limited (all reasonable configurations converge to the same F1).

```bash
# Full sweep on a registered dataset. This repository keeps raw runs under
# results/experiments and derived artifacts under results/exports and
# results/reports; the defaults are the same paths without the intermediate
# directory, so pass --root/--out/--exports explicitly.
python -m prebenchmark generate --experiment-id exp-sweep-02 --dataset satimage \
  --root results/experiments
python -m prebenchmark run --experiment-id exp-sweep-02 --dataset satimage \
  --root results/experiments --multi-seed --earlyStop --class-weight none
python -m prebenchmark export --experiment-id exp-sweep-02 \
  --root results/experiments --out results/exports
python -m prebenchmark analyze --experiment-id exp-sweep-02 \
  --exports results/exports --out results/reports
```

`wine_quality` keeps the historical UCI download and dimensions, so existing
experiments remain reproducible. The other datasets require network access only
on their first load; afterwards they are served from the cache.

## Statistical analysis report

The report lists every configuration in the export and compares their metrics
visually and inferentially. Report text, table headers and figure labels are in
Spanish; the code and CSV column names stay in English:

- **Ranking and distributions** — mean F1 macro per configuration with sample
  standard deviation, pooled and per-`class_weight` histograms.
- **Marginal views** — boxplots and Kruskal-Wallis tests by activation,
  optimizer, learning rate, initialization and `class_weight`.
- **Convergence** — scatter plots and histograms of `best_epoch`,
  `epochs_executed` and `time_to_best_epoch`, plus validation-loss curves for
  the top configurations (mean ± std across seeds, individual seeds in grey).
- **Activation × optimizer heatmap** — mean F1 per cell, faceted by
  `class_weight`, with a shared colour scale.
- **Inferential comparison** — each configuration against a reference
  (empirical best by mean F1, or `--reference-config-id`), reporting the
  difference, its 95 % Welch interval, Welch and Mann-Whitney p-values, and
  Cliff's delta with Romano effect-size labels.

Convergence metrics are derived from the per-epoch history, not guessed:

| Metric | Definition |
| --- | --- |
| `best_epoch` | epoch with the lowest `val_loss` (consistent with `restore_best_weights`) |
| `epochs_executed` | epochs actually trained before early stopping or the epoch cap |
| `time_to_best_epoch` | cumulative `epoch_time` up to `best_epoch` |

`class_weight` is absent from `master_*.csv`; the analyzer joins it from
`configs_*.csv` by `config_id`. The comparison unit is the configuration
**without the seed** (`group_key`), so the five seeds of a configuration are
treated as repetitions rather than as five different configurations.

With five seeds per configuration the inferential tests have low power. The
report presents p-values as evidence and recommends reading them together with
the confidence intervals and effect sizes.

Useful flags: `--no-figures` (tables and statistics only), `--top-curves N`
(number of configurations in the convergence-curve panel),
`--reference-config-id` (accepts a `group_key` or any member `config_id`),
`--out` (reports root) and `--pdf` (also print the HTML to PDF).

### PDF output

`--pdf` prints the generated HTML to a PDF with Chrome or Edge in headless
mode. The PDF is written next to the HTML with the same base name:

```bash
python -m prebenchmark analyze --experiment-id exp-001 --pdf
# results/reports/exp-001/analysis_<timestamp>.html
# results/reports/exp-001/analysis_<timestamp>.pdf
```

The browser is auto-detected on Windows, macOS and Linux. If it is installed in
a non-standard location, pass `--browser "C:\path\to\chrome.exe"`. The PDF uses
A4 portrait, repeats table headers across pages and splits the wide
configuration tables into quality (1a) and convergence (1b) halves so nothing is
clipped.

An existing HTML can also be printed with the generic PowerShell helper:

```powershell
.\generar-pdf.ps1 -Html "results\reports\exp-001\analysis_<timestamp>.html" `
                  -Pdf  "results\reports\exp-001\analysis_<timestamp>.pdf"
```

## Persistence location

Results are written under the store root (`results/` by default):

```
results/
  <experiment-id>/
    <run-id>/
      metadata.json      # run record: config snapshot, dataset version, env, state
      epochs.csv         # per-epoch train/val history
    configs.jsonl        # configuration registry (one per config_id; orphaned entries are purged on delete)
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

## Code quality

Development tools live in `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt

ruff check prebenchmark tests   # lint
ruff format prebenchmark tests  # format
mypy                                         # type checking (prebenchmark + server)
```

Configuration for ruff and mypy is centralized in `pyproject.toml`. The file has
no `[project]` or `[build-system]` section, so it configures tools only; the
package is still executed from the repository root (`python -m prebenchmark`)
rather than installed.

## Web UI (single-run and suite comparison)

```bash
pip install -r requirements-server.txt
python -m uvicorn server.app:app --reload
# http://127.0.0.1:8000
```

The UI has two implemented tabs: a **single-configuration run** (real training,
live loss curve over SSE, validation metrics, cancel button, optional comparison
of up to three runs with per-row removal, full `metadata.json` viewer) and a
**suite comparison** tab that reads the exported evidence and serves the generated
HTML/PDF reports. A compact **history** button above the loss curve opens a modal
with every run persisted in `results/ui-runs`, so the traceability survives server
restarts; each row can be expanded into its full metadata and rebuilt loss curve,
or permanently deleted after an explicit confirmation.

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | worker count, isolated store root, experiment id |
| GET | `/api/options` | curated dataset menu, the four free dimensions, protocol defaults |
| GET | `/api/datasets/{name}` | real dimensions and class imbalance (loads/caches on demand) |
| POST | `/api/validate` | `space.validate_config` without running |
| POST | `/api/runs` | enqueue one run (202 + `job_id`) |
| GET | `/api/runs/{id}` | job snapshot (in-memory) |
| POST | `/api/runs/{id}/cancel` | cooperative cancel; the run ends as `INCOMPLETE` |
| GET | `/api/runs/{id}/events` | SSE stream: `status`, `epoch`, `done`, `failed` |
| GET | `/api/history` | persisted runs from disk, newest first (`limit` ≤ 500) |
| GET | `/api/history/{run_id}` | one persisted run: metadata + `epochs.csv` rows |
| DELETE | `/api/history/{run_id}` | permanently delete a run; purge its config only if unreferenced |
| GET | `/api/evidence` | family statistics and failure modes from `results/exports` |
| GET | `/api/reports/{experiment}` | latest generated HTML report |
| GET | `/api/reports/{experiment}/pdf` | latest generated PDF report |

Design decisions the skeleton enforces:

- **One worker.** `runner` seeds the process-global NumPy/torch RNGs, so
  concurrent runs in the same process would break reproducibility. `JobRegistry`
  refuses `max_workers != 1`.
- **Isolated store.** UI runs go to `results/ui-runs/ui-exploratory/` by default,
  never to the suite store. Override with `PREBENCHMARK_UI_STORE` and
  `PREBENCHMARK_UI_EXPERIMENT`.
- **Test set reserved.** The response exposes `validation_metrics` (now including
  macro F1, precision, recall, balanced accuracy and confusion matrix at the best
  epoch). `final_metrics` is returned only when the request sets
  `include_test=true`, and is labelled as test.
- **Early stopping on by default** in the UI, matching the sweeps; without it a
  run can consume all 500 epochs.
- **Evidence, not hardcoded lists.** The "problematic family" warnings are
  computed from the exported runs and classify the failure mode from the executed
  epochs (early collapse vs. exhausting the epoch cap).
- **Observer hook.** `runner.run_experiment` accepts `on_epoch`, called with each
  `epochs.csv`-shaped record. A failing observer never aborts a run.
- **Cooperative cancel.** `runner.run_experiment` accepts `should_stop`, checked
  at the start of every epoch. A cancelled run is recorded as `INCOMPLETE`
  (already part of the technical-state vocabulary) and excluded from aggregates
  and reports; its best-epoch metrics are still available for inspection.

`PREBENCHMARK_TORCH_THREADS` optionally pins `torch.set_num_threads` when the
process shares the machine with other loads.

## Tests

```bash
python -m pytest -q
```

The suite runs fully offline (in-memory datasets; no network required).

## Docker (local deployment)

A CPU-only image is provided for running the UI or the CLI on another machine
without installing Python or PyTorch:

```bash
docker compose up --build
# http://127.0.0.1:8000
```

`results/` and `.cache/` are bind-mounted from the host, so runs, exports,
reports and dataset caches survive container recreation. The compose file binds
to loopback because the API has no authentication. See
[DEPLOYMENT.md](DEPLOYMENT.md) for persistence details, CLI usage, Linux
permissions, and the known limits (single worker, no PDF generation in the
image).

## Desktop app (Electron + Python sidecar)

`desktop/` packages the same FastAPI UI and training engine as a cross-platform
desktop application. Electron owns the window and process lifecycle; a
PyInstaller sidecar serves the existing API on loopback and works fully offline
(datasets and reference exports are bundled). Data lives in the per-user
application directory, never in the installation folder.

```bash
python desktop/backend/stage_datasets.py
python -m PyInstaller desktop/backend/prebenchmark-sidecar.spec \
  --noconfirm --distpath desktop/backend/dist --workpath desktop/backend/build
cd desktop && npm install && npm start
```

Installers (`nsis`, `dmg`, `AppImage`, `deb`) are built per OS with
`npm run dist:win|mac|linux`. See [desktop/README.md](desktop/README.md) for the
architecture, data locations, code signing and the GitHub Actions workflow.

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
