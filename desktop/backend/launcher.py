"""Entry point for the packaged Python sidecar.

The Electron shell spawns this executable with ``--data-dir`` and ``--port``.
It owns the process working directory (the data dir), seeds the offline
datasets, starts uvicorn on loopback and shuts down when stdin closes or a
``shutdown`` line arrives.

Why a sidecar and not an in-process import: Electron runs Node, and the
training engine is Python. The sidecar keeps the existing FastAPI app and the
whole ``prebenchmark`` package untouched; the shell only owns the window.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from pathlib import Path

SEED_VERSION = "1"


def resource_dir() -> Path:
    """Directory that holds bundled data, both frozen and from source."""
    if getattr(sys, "frozen", False):
        # PyInstaller 6 onedir places data under _internal (sys._MEIPASS).
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    # desktop/backend/launcher.py -> repository root
    return Path(__file__).resolve().parents[2]


def _copy_missing_files(source: Path, target: Path) -> int:
    if not source.is_dir():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if not destination.exists():
            shutil.copy2(item, destination)
            copied += 1
    return copied


def seed_sources(resources: Path) -> tuple[Path, Path]:
    """Return the dataset and evidence sources for frozen and source runs."""
    if getattr(sys, "frozen", False):
        return resources / "datasets", resources / "seed-results"
    # From source the repository already holds both inputs.
    return resources / ".cache" / "prebenchmark", resources / "results"


def seed_datasets(data_dir: Path, resources: Path) -> int:
    """Copy the bundled canonical CSVs so the app works with no network."""
    datasets_source, _ = seed_sources(resources)
    return _copy_missing_files(datasets_source, data_dir / ".cache" / "prebenchmark")


def seed_results(data_dir: Path, resources: Path) -> None:
    """Seed the exported evidence once, without clobbering student runs."""
    _, source = seed_sources(resources)
    if not source.is_dir():
        return
    marker = data_dir / "results" / ".seed-version"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == SEED_VERSION:
        return
    for name in ("exports", "reports"):
        origin = source / name
        if origin.is_dir():
            shutil.copytree(origin, data_dir / "results" / name, dirs_exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(SEED_VERSION, encoding="utf-8")


def _watch_stdin(server: object) -> None:
    """Treat stdin EOF or an explicit line as a cooperative shutdown request."""
    try:
        for line in sys.stdin:
            if line.strip().lower() in {"shutdown", "quit", "exit"}:
                break
    except (OSError, ValueError):
        pass
    server.should_exit = True  # type: ignore[attr-defined]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebenchmark desktop sidecar")
    parser.add_argument("--data-dir", required=True, help="writable per-user data directory")
    parser.add_argument("--port", type=int, required=True, help="loopback port to bind")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # The store, exports and dataset cache are relative paths in the app.
    # Owning cwd here keeps the domain code unchanged and the data portable.
    os.chdir(data_dir)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PREBENCHMARK_UI_STORE", "results/ui-runs")
    os.environ.setdefault("PREBENCHMARK_EXPORTS", "results/exports")
    os.environ.setdefault("PREBENCHMARK_REPORTS", "results/reports")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    resources = resource_dir()
    if not getattr(sys, "frozen", False) and str(resources) not in sys.path:
        sys.path.insert(0, str(resources))

    seeded = seed_datasets(data_dir, resources)
    seed_results(data_dir, resources)

    import uvicorn

    from server.app import app

    print(
        f"sidecar: data_dir={data_dir} datasets_seeded={seeded} "
        f"host={args.host} port={args.port}",
        flush=True,
    )

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    threading.Thread(target=_watch_stdin, args=(server,), daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
