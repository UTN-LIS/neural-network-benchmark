# Local deployment with Docker

This image is meant for **one instance per machine**: each researcher runs the app
locally and keeps their own results. It is not a multi-user server.

## Quick start

```bash
docker compose up --build
# http://127.0.0.1:8000
```

The first build downloads the CPU build of PyTorch and takes a few minutes. The
resulting image is about 2 GB (torch, scipy and matplotlib dominate). Later
starts are fast. `docker compose down` stops the container; the data stays on
the host.

To use another port:

```bash
PREBENCHMARK_PORT=8080 docker compose up
```

## What persists, and where

A container is disposable. Persistence comes from the two bind mounts in
`docker-compose.yml`:

| Host path | Container path | Contents |
| --- | --- | --- |
| `./results` | `/app/results` | Runs (`metadata.json`, `epochs.csv`), `configs.jsonl`, `decisions.jsonl`, exports and reports |
| `./.cache` | `/app/.cache` | Canonical dataset CSVs (sha256 fingerprinted) |

`docker compose down -v` does **not** delete bind mounts, but deleting those host
directories does. To back up the evidence, copy `./results`; `./.cache` can be
regenerated. The cache path matches the CLI default (`.cache/prebenchmark`), so
host and container share the same downloaded datasets.

The UI store defaults to `results/ui-runs/ui-exploratory`, separate from the suite
store in `results/experiments`. Override it with `PREBENCHMARK_UI_STORE` and
`PREBENCHMARK_UI_EXPERIMENT`.

## Datasets need network access once

`wine_quality` (UCI) and the OpenML datasets are downloaded on first use and then
served from `./cache`. If the machine is offline, pre-populate `.cache` or use a
dataset that is already cached. A failed download surfaces as HTTP 502 in the UI
and as an error in the CLI.

## CLI inside the container

The same image runs the headless pipeline:

```bash
docker compose run --rm app python -m prebenchmark generate \
  --experiment-id exp-001 --limit 36

docker compose run --rm app python -m prebenchmark run \
  --experiment-id exp-001 --limit 36 --max-epochs 200
```

`docker compose run` starts a separate container that shares the same mounts. Do
**not** run a CLI sweep while the server container is training: both write to
`./results`, and `configs.jsonl` registration is read-then-append, so concurrent
processes can duplicate records. Stop the server first, or use a different
`--root`.

## Updating the image

```bash
docker compose build --pull
docker compose up -d
```

Finish or cancel any in-flight run first. Live job state lives in memory
(`server/jobs.py`), so recreating the container loses the *in-flight* job
registry; every completed run remains on disk and is still listed by the
history modal (`GET /api/history`).

## Linux permissions

The image runs as uid 1000 (`app`). On Linux, create the mount points first and
give them to that uid:

```bash
mkdir -p results .cache
sudo chown -R 1000:1000 results .cache
```

Windows and macOS with Docker Desktop ignore this: bind mounts are mapped through
the desktop file-sharing layer.

## Exposing it beyond localhost

The compose file binds to `127.0.0.1` on purpose. **The API has no
authentication**: anyone who can reach it can launch and cancel runs and read
every persisted report. On a shared or public network, put it behind a reverse
proxy with authentication and TLS, or use a VPN, and change the port mapping to
`0.0.0.0`.

Two constraints if you expose it:

- **One run at a time.** `JobRegistry` refuses `max_workers != 1` because
  `runner` seeds process-global NumPy/torch RNGs. Multiple users queue behind
  each other. Real parallelism needs separate processes, not a bigger worker
  pool.
- **Keep it at the domain root.** The UI uses absolute `/api/...` and `/static/...`
  paths and `EventSource("/api/runs/...")`. Serving it under a subpath such as
  `https://host/prebenchmark/` breaks both.

## Known limits of this image

- **No database, by design.** The store is append-only filesystem state; see
  `prebenchmark/store.py`. A full suite (180 runs) is ~6 MB, so a volume is
  cheaper and safer than a database or a hosted backend.
- **No PDF generation.** `python -m prebenchmark analyze --pdf` needs Chrome or
  Edge in headless mode, which this slim image does not include. Generate the
  PDF on the host with the same command, or extend the image with Chromium and
  pass `--browser`. HTML reports work everywhere.
- **No `--reload`.** The server would restart on file changes and kill the
  running training job.
- **CPU only.** The image installs `torch==2.14.1+cpu`. For CUDA, build a
  different image with the matching CUDA index and a GPU runtime; the default
  CPU image does not see the host GPU.

## Troubleshooting

- **Port already in use**: set `PREBENCHMARK_PORT` to a free port.
- **Build fails resolving torch**: the pinned version may not exist on the CPU
  index for this Python. Check `https://download.pytorch.org/whl/cpu/torch/` and
  update `TORCH_VERSION` in `docker-compose.yml`.
- **`/api/evidence` is empty**: the suite comparison tab reads
  `results/exports` and `results/reports`. A fresh clone already contains the
  tracked exports/reports; if you deleted them, regenerate them with
  `export` and `analyze`.
- **`permission denied` writing results (Linux)**: apply the uid 1000 ownership
  step above.
