# Exploration: prebenchmark-infra

> Phase: sdd-explore | Date: 2026-10-08 | Artifact store: openspec
> Input: user requirements doc (`prebenchmark-requirements.md`, Spanish — authoritative)

## Current State

Repository is a script-based experiment suite. Everything relevant was verified by reading real code:

- **Entry points**: three monolithic top-level scripts under `Experimentos/` —
  `wine.py` (194 LOC), `banknote.py` (273 LOC), `wine_quality.py` (301 LOC).
  No package layout, no `pyproject.toml`, no CLI, no `if __name__ == "__main__"` guard.
  Importing any script triggers data load + full training (module-level side effects).
- **Shared pipeline (duplicated in all 3 scripts)**: seed (np + torch, `SEED = 42`) →
  load dataset → `train_test_split(test_size=0.2, random_state=SEED, stratify=y)` →
  `StandardScaler` → tensors → hardcoded `MLP` class → He/Kaiming init loop →
  `CrossEntropyLoss` + `Adam(lr=0.001)` → fixed-epoch **full-batch** training →
  test-set eval → sklearn metrics → `print`.
- **Datasets**: `wine` via `sklearn.datasets.load_wine` (bundled, offline-safe);
  `banknote` and `wine_quality` download UCI CSVs at module scope.
- **Metrics**: `banknote.py` / `wine_quality.py` compute accuracy, balanced accuracy,
  precision/recall/F1 macro, confusion matrix; `wine.py` computes accuracy only.
  All are computed **once, on the test set, at the end**. No validation split exists anywhere.
- **Per-epoch data**: only `loss` is printed every 10 epochs. Nothing is persisted;
  all results go to stdout. No files, no store, no history.
- **Parameterization**: activation (ReLU), optimizer (Adam), lr (0.001), init (He),
  architecture, epochs are hardcoded constants. `BATCH_SIZE = 32` exists in `wine.py`
  but is unused (comment admits full-batch training); no `DataLoader` anywhere.
- **Device**: implicitly CPU — tensors are never moved, no CUDA code.
- **Docs**: `README.md` is 2 lines. No `AGENTS.md`/`agents.md`, no design docs,
  **no documented baseline configuration** anywhere. `openspec/` bootstrapped by
  sdd-init (config.yaml + empty dirs only).
- **Tests**: none. No pytest/unittest config, no test deps in `requirements.txt`
  (a pinned freeze) or `.venv`. No linter/CI.
- **Git state**: `Experimentos/wine_quality.py` has uncommitted changes on `main`
  (`EPOCHS 100 → 1000`, one added print). MUST remain untouched by this change.

### What the requirements assume vs what exists

| Requirement area | Exists today? | Notes |
|---|---|---|
| Explicit parameter space (act/opt/lr/init) | ❌ | All hardcoded; only ReLU/Adam/0.001/He exercised |
| Config generation + VALID/WARNING/INVALID | ❌ | Nothing |
| Run identity (config/run/repetition-group IDs) | ❌ | Nothing |
| Per-epoch metric persistence | ❌ | Loss printed only; no history |
| Final metric suite persistence | ❌ | Printed only |
| Validation-set metrics | ⚠️ partial | No val split exists; requirement says "when a validation set exists" |
| Structured store (historical, no overwrite) | ❌ | Nothing |
| Baseline comparator + deltas | ❌ | Nothing; **no baseline documented** |
| Post-run classification + decision log | ❌ | Nothing |
| Repetition grouping + statistics | ❌ | Nothing |
| Train/eval/metric logic to reuse | ✅ | Duplicated inline in the 3 scripts (this is the reusable core) |
| SEED=42 reproducibility | ✅ | np + torch; `random_state=SEED` in sklearn splits (matches req `random_state=42`) |
| Automated tests (AC 12) | ❌ | Test runner entirely absent |

### Inconsistencies between current code and requirements

1. **No validation set** anywhere → §6.2 val loss/accuracy and overfitting analysis
   need an optional val split that does not exist. Design must make it configurable.
2. **No batching** → batch size is a recordable/explorable parameter (§4.5), but
   current training is full-batch. Introducing `DataLoader` changes training semantics
   (old prints are not comparable anyway — nothing was persisted — but this must be an
   explicit design decision).
3. **Only `wine.py` has `BATCH_SIZE`, unused** → config field would be meaningless
   without mini-batch support.
4. **`wine.py` metric suite is incomplete** (accuracy only) vs §6.1.
5. **No device/hardware capture** (§6.3) — code never touches CUDA.
6. **Fixed epoch count only** — no epochs-executed vs max-epochs distinction, no
   configurable convergence criterion (§6.2 requires it to be explicit and recorded).
7. **Module-level side effects** block reuse, testing (AC 12) and safe import.
8. **Tension to resolve explicitly**: §13 says "reuse existing components, avoid
   duplicating logic", but AC 13 says run the flow "without unnecessary changes to the
   other scripts", and `wine_quality.py` has protected uncommitted changes.
   Resolution: extract the shared pipeline into parameterized functions in a new
   package (one future implementation), keep the three legacy scripts **untouched**
   in this change; thin-wrapper refactoring of scripts is deferred out of scope.

## Affected Areas

- `Experimentos/*.py` — source of the reusable pipeline logic (read/extract only;
  NOT modified in this change, per AC 13 + protected uncommitted changes)
- `requirements.txt` — will need `pytest` (dev/test dependency for AC 12)
- `.gitignore` — decision needed on whether `results/` is ignored or committed
- `openspec/changes/prebenchmark-infra/` — all new artifacts live here
- **New** (proposed): `prebenchmark/` package (config/data/model/train/metrics/
  store/compare/aggregate/runner) + `tests/` + `results/` output tree

## Approaches

### 1. Package-first extraction (recommended)

Build a `prebenchmark/` package: generalized versions of the pipeline now in the
scripts (data loading, MLP construction, init, training loop with per-epoch capture,
metrics) plus the new infrastructure (parameter space, validation, identity, store,
comparator, aggregation, runner entry point). Legacy scripts stay untouched.

- Pros: satisfies "no parallel solution" in spirit (single implementation going
  forward); import-safe, unit-testable; zero risk to `wine_quality.py`; matches the
  §15 incremental order; slices cleanly under the 400-line PR budget.
- Cons: temporarily the pipeline logic exists twice (scripts + package); "reuse" is
  by extraction, not by import — must be stated plainly in the proposal.
- Effort: High (but phased)

### 2. Layer-on-top framework (rejected)

New package reimplements training from scratch without extracting script logic,
scripts untouched.

- Pros: fastest, zero script risk.
- Cons: pure duplication — directly violates §13 ("reuse existing components,
  avoid duplicating logic between scripts"); a "parallel solution" the user forbade.
- Effort: Medium

### 3. Extract-and-refactor scripts into thin wrappers

Same as (1) plus rewriting the three scripts to call the package.

- Pros: eliminates duplication completely; strongest reading of §13.
- Cons: touches protected `wine_quality.py` (uncommitted changes must be carried);
  enlarges scope beyond AC 13's "no unnecessary changes"; higher review cost.
- Effort: High

### Persistence format comparison

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| CSV only | pandas-native | nested config awkward; per-run metadata/epoch history collide; overwrite risk | Weak alone |
| JSON/JSONL only | nested-friendly, append-only JSONL | epoch tables less convenient for pandas analysis | Good for run records |
| SQLite (stdlib) | queryable, single file, no new deps | less transparent/diffable, more code, overkill at this scale | Defer |
| **Hybrid: per-run directory (JSON metadata + `epochs.csv`) + aggregate CSV exports** | never overwrites by construction (run_id = directory), transparent, portable, pandas-friendly | needs a small loader to build aggregate tables | **Recommended** |

Rationale: §7 asks to "evaluate" the alternatives and prioritize simple, transparent,
portable, pandas-analyzable storage. Per-run directories make "historical, never
overwrite" (AC 6) structurally true; configs stored once under stable `config_id`;
final aggregates/exports as CSV. SQLite documented as a deferred option.

## Recommendation

**Approach 1 (package-first extraction), hybrid persistence, phased delivery.**

- New package `prebenchmark/` with modules matching the §15 order:
  `space` (parameter space + generation + VALID/WARNING/INVALID validation) →
  `store` (identity, per-run persistence) → `runner` (integration with training,
  per-epoch capture) → `compare` (baseline deltas, controlled comparisons) →
  `aggregate` (repetition statistics) → `export` (tables/CSV).
- First slice: **data model + config generation/validation + pytest bootstrap** —
  no execution side effects, fully unit-testable, zero risk, satisfies §15 items 1–2
  and starts AC 12.
- Estimated total: ~2000+ changed lines → **exceeds the 400-line review budget →
  chained PRs required** (delivery strategy `ask-always`: confirm slicing with the
  user at sdd-tasks).
- Test strategy: introduce `pytest` as the project's first test runner (required by
  AC 12; `strict_tdd: false` per openspec config means tests are added alongside,
  not RED-GREEN enforced). Tests target the pure package code; no script imports.

### Key uncertainties surfaced (for proposal/design)

1. **Baseline**: none documented anywhere → propose explicitly: the repo's de-facto
   historical configuration **per dataset** (ReLU + Adam + lr 0.001 + He init +
   CrossEntropy + existing architecture), stated as "the only configuration ever run
   in this project". Assumptions must be written into the proposal before adoption.
2. **Convergence criteria**: no universal criterion; must be an optional,
   config-declared, recorded field with default = disabled (fixed max epochs). No
   invented thresholds (§6.2, §11).
3. **Validation split**: currently absent. Recommend a configurable `val_size`
   (recorded in config); when 0, val metrics are recorded as N/A. Default value is a
   proposal-stage decision — must not silently change the 0.8/0.2 split semantics.
4. **Batching**: recommend real mini-batch support via `DataLoader` with `batch_size`
   in config (otherwise the parameter is unregistrable in any meaningful sense);
   flag that this changes training behavior vs. the legacy full-batch scripts.
5. **Post-run classification thresholds**: VIABLE/INTERESTING/UNSTABLE etc. must be
   rule-driven with configurable, documented rules; defaults proposed with
   justification, never hardcoded arbitrary cut-offs (§9.2, §11).
6. **`results/` in git**: recommend `.gitignore` for raw run output + explicit export
   snapshots if the user wants committed summaries (user decision).
7. **Repetition counts / statistics sufficiency**: no arbitrary minimums; report
   "insufficient repetitions" flags with a configurable minimum (default proposed, not
   imposed).

## Risks

- **Scope size**: full change is large; without chained-PR slicing the 400-line
  budget will be blown in one PR.
- **Protected file**: any edit to `Experimentos/wine_quality.py` risks losing the
  user's uncommitted `EPOCHS=1000` change — keep it out of this change entirely.
- **Semantics drift**: introducing batching/validation/device code means new runs are
  not comparable to the legacy prints; acceptable (nothing was persisted) but must be
  documented.
- **Network-dependent datasets**: `banknote`/`wine_quality` download at module scope;
  the smoke test (AC 13) should prefer bundled `load_wine` for offline reliability.
- **No test runner today**: bootstrapping pytest is a new dependency and a process
  change; must be agreed in the proposal.
- **Comparability guard**: comparator must not compare across different
  datasets/splits/architectures without warnings (§8.2, §13) — a correctness trap.

## Ready for Proposal

**Yes.** Requirements are detailed and verified against real code. The orchestrator
should confirm with the user: (a) baseline proposal above, (b) pytest bootstrap,
(c) `val_size`/batching defaults as open design decisions, (d) chained-PR slicing
(ask-always strategy), and (e) that legacy scripts remain untouched in this change.
