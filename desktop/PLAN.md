# Desktop delivery plan (Electron + Python sidecar)

## Goal

Ship the existing benchmark UI and training engine as a downloadable, offline
desktop application for Windows, macOS and Linux, installable by students with
no Python, Node or Docker knowledge.

## Decision

Electron owns the window and process lifecycle. A PyInstaller-packaged Python
sidecar runs the unchanged FastAPI app on a dynamic loopback port. Electron waits
for `/api/health`, loads the URL in a `BrowserWindow`, and requests a graceful
shutdown on quit.

Rejected alternatives:

- **Rewrite in JS**: loses PyTorch/scikit-learn and the whole pipeline.
- **Browser + launcher**: lowest effort, but does not meet the "own window"
  requirement.
- **pywebview**: good Windows-only fit, but the target is now cross-platform and
  Electron gives a consistent Chromium renderer.

## Phases

| Phase | Deliverable | Status |
| --- | --- | --- |
| P0. Packaging spike | Frozen sidecar imports torch/sklearn/matplotlib, serves health and trains one epoch | done and verified |
| P1. Shell | Electron main process, splash, dynamic port, health wait, graceful shutdown, single instance | done and verified |
| P2. Windows build | NSIS installer + portable ZIP, packaged app launched from the ZIP | done on Windows |
| P3. Cross-platform | macOS DMG/ZIP and Linux AppImage/deb configs + GitHub Actions matrix | config ready; artifacts must be built on each OS or in CI |
| P4. Distribution | Code signing, notarization, classroom distribution | documented, out of scope for today |

## Verification results (Windows)

| Check | Result |
| --- | --- |
| Frozen sidecar bundle | 627 MB, health OK |
| Real training from the frozen sidecar | 1 epoch, `COMPLETED`, run persisted |
| Electron dev shell | window loaded, UI HTTP 200, graceful shutdown |
| Packaged app (`win-unpacked`) | sidecar resolved from `resources/backend`, health OK |
| NSIS installer | 294 MB, per-user, unsigned |
| Portable ZIP | 371 MB, contains app, asar and sidecar |
| Installed footprint | 984 MB |

## Verification matrix

| Check | Where |
| --- | --- |
| Frozen sidecar health | `python desktop/scripts/smoke_sidecar.py` |
| One real epoch through the frozen sidecar | same script |
| Datasets seeded offline | smoke script temp data dir |
| Electron window loads the UI | `npm start` from `desktop/` |
| Packaged resources path, asar and sidecar | run the unpacked ZIP build |
| Installer artifact produced | `npm run dist:win` |
| macOS/Linux artifacts | GitHub Actions matrix |

## Risks

1. **PyInstaller + torch** is the main technical risk; P0 exists to fail fast.
2. **Unsigned binaries** trigger SmartScreen and Gatekeeper. Functional, but a
   distribution problem that needs certificates or IT whitelisting.
3. **Bundle size**: about 1.2 GB installed. Distribute by USB or local network.
4. **Cross-compilation is not possible** for the Python sidecar; each OS builds
   its own artifact in CI.

## Commands

```bash
# Backend (from repo root)
python desktop/backend/stage_datasets.py
python -m PyInstaller desktop/backend/prebenchmark-sidecar.spec \
  --noconfirm --distpath desktop/backend/dist --workpath desktop/backend/build
python desktop/scripts/smoke_sidecar.py

# Desktop shell
cd desktop
npm install
npm start

# Installers
npm run dist:win
npm run dist:mac
npm run dist:linux
```
