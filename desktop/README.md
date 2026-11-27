# Desktop app (Electron + Python sidecar)

This directory wraps the existing FastAPI UI and the `prebenchmark` engine in a
desktop application. It is an additional distribution channel, not a rewrite:
the UI, the API and the results store are exactly the same as in the browser
build.

## Architecture

```
Electron main process
  ├── chooses a free loopback port
  ├── spawns resources/backend/prebenchmark-sidecar
  ├── waits for GET /api/health
  ├── opens a BrowserWindow on http://127.0.0.1:<port>
  └── writes "shutdown" to the sidecar's stdin on quit

prebenchmark-sidecar (PyInstaller one-dir)
  ├── receives --data-dir and --port
  ├── chdir to the per-user data directory
  ├── seeds the offline datasets and the reference exports
  └── runs uvicorn in the main thread
```

Electron owns the window and process lifecycle; Python owns training and
persistence. There is no JS rewrite of the domain and no Node/Python bridge
beyond the local HTTP API.

## Why the data directory matters

The desktop app cannot write to the installation directory. The sidecar changes
its working directory to a per-user folder before importing the app, which keeps
the existing relative paths (`results/`, `.cache/`) valid without touching the
domain code.

Typical locations (Electron uses the application name):

| OS | Data directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Neural Network Benchmark\benchmark` |
| macOS | `~/Library/Application Support/Neural Network Benchmark/benchmark` |
| Linux | `~/.config/Neural Network Benchmark/benchmark` |

Logs go to the sibling `logs/` folder (`backend.log`, `desktop.log`).

Back up or collect student work by copying the `benchmark/results` directory.

## Offline behavior

The sidecar bundle contains:

- the six canonical dataset CSVs (4.2 MB), copied to `.cache/prebenchmark` on
  first run;
- the exported suites and the generated HTML/PDF reports currently in the
  repository, copied once into `results/exports` and `results/reports`.

No runtime request leaves the machine. Datasets that were removed from the UI
menu but remain registered are still bundled, so the API stays complete.

## Prerequisites

- Python 3.13 with `requirements.txt` and `desktop/backend/requirements-build.txt`
  installed in the active environment.
- Node.js 20+ (tested with Node 24) and npm.
- The canonical datasets present in `.cache/prebenchmark` (the staging script
  downloads anything missing; this is the only network step of the build).

## Development

From the repository root:

```bash
python desktop/backend/stage_datasets.py
python -m PyInstaller desktop/backend/prebenchmark-sidecar.spec \
  --noconfirm --distpath desktop/backend/dist --workpath desktop/backend/build
```

Then:

```bash
cd desktop
npm install
npm start
```

`npm start` runs Electron against `backend/dist/prebenchmark-sidecar`. The
production build copies that directory into the app resources.

## Packaging

Build the sidecar first, then the installer:

```bash
cd desktop
npm run build:backend
npm run dist:win     # NSIS installer + portable ZIP
npm run dist:mac     # DMG + ZIP
npm run dist:linux   # AppImage + deb
```

Artifacts land in `desktop/release/`.

**Build each target on its own OS.** PyInstaller does not cross-compile, and a
macOS sidecar cannot be produced on Windows. The GitHub Actions workflow
`.github/workflows/desktop.yml` builds all three platforms in parallel.

## Code signing

The current configuration produces unsigned artifacts:

- Windows shows SmartScreen warnings and some antivirus engines flag
  PyInstaller bootloaders. Sign the installer with an Authenticode certificate
  for classroom distribution.
- macOS sets `identity: null`; Gatekeeper blocks unsigned apps unless the user
  right-clicks and chooses Open, or the app is signed and notarized with a
  Developer ID certificate.

Signing is a distribution concern, not a functional one: the app runs offline
once the OS allows it.

### Running an unsigned build

- **Windows**: SmartScreen shows "Windows protected your PC". Choose *More info*
  → *Run anyway*. The installer does not need administrator rights.
- **macOS**: Gatekeeper blocks the first launch. Right-click the app → *Open* →
  *Open*, or remove the quarantine flag:
  `xattr -dr com.apple.quarantine "/Applications/Neural Network Benchmark.app"`.
- **Linux**: AppImage needs `libfuse2` on some distributions
  (`sudo apt install libfuse2`). Make it executable with `chmod +x` if the file
  manager does not do it.

## Limits

- One run at a time. The engine seeds process-global RNGs, so the job registry
  enforces a single worker.
- Closing the window stops the sidecar. A run in progress is cancelled; completed
  runs remain on disk.
- The renderer is locked down: `contextIsolation`, `sandbox`, no Node integration
  and no preload. The local API is the only surface.
- The installer is large (~1.2 GB installed) because PyTorch is bundled.

## Troubleshooting

- **"No se encontró el backend empaquetado"**: run `npm run build:backend` from
  `desktop/`.
- **Backend timeout**: read `logs/backend.log` in the data directory. A missing
  dataset or a failed import is reported there.
- **Port conflicts do not happen**: the port is chosen dynamically on
  `127.0.0.1` for every launch.
- **Linux/macOS permission denied**: confirm the sidecar is executable; the
  `afterPack` hook sets mode `0755` when packaging.
