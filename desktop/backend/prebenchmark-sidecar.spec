# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the prebenchmark desktop sidecar.

Build from ``desktop`` with:

    python -m PyInstaller backend/prebenchmark-sidecar.spec \
        --noconfirm --distpath backend/dist --workpath backend/build

The result is a one-directory bundle. One-file is intentionally avoided: a
~1 GB executable would decompress on every launch and add several seconds to
startup.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821 - injected by PyInstaller
REPO_ROOT = SPEC_DIR.parents[1]

DATASETS_DIR = REPO_ROOT / ".cache" / "prebenchmark"
EXPORTS_DIR = REPO_ROOT / "results" / "exports"
REPORTS_DIR = REPO_ROOT / "results" / "reports"

for required in (DATASETS_DIR, REPO_ROOT / "server" / "static"):
    if not required.exists():
        raise SystemExit(
            f"missing build input: {required}\n"
            "Run `python desktop/backend/stage_datasets.py` from the repo root first."
        )

datas = [
    (str(REPO_ROOT / "server" / "static"), "server/static"),
    (str(DATASETS_DIR), "datasets"),
]
if EXPORTS_DIR.is_dir():
    datas.append((str(EXPORTS_DIR), "seed-results/exports"))
if REPORTS_DIR.is_dir():
    datas.append((str(REPORTS_DIR), "seed-results/reports"))

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("server")
    + collect_submodules("prebenchmark")
    + ["matplotlib.backends.backend_agg"]
)

a = Analysis(  # noqa: F821 - injected by PyInstaller
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "mypy", "ruff", "tkinter", "IPython", "notebook"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="prebenchmark-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="prebenchmark-sidecar",
)
