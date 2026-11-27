"""End-to-end smoke test for the packaged sidecar.

Starts the frozen sidecar, waits for /api/health, runs one real epoch through the
HTTP API and verifies the result. This is the check that proves torch, sklearn
and the FastAPI app work outside the virtual environment.

    python desktop/scripts/smoke_sidecar.py [--exe PATH] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

READY_TIMEOUT = 180
RUN_TIMEOUT = 600


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers, method=method), timeout=10
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        try:
            if request("GET", url).get("status") == "ok":
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"sidecar did not become healthy in {timeout}s")


def default_exe() -> Path:
    name = "prebenchmark-sidecar.exe" if os.name == "nt" else "prebenchmark-sidecar"
    return Path(__file__).resolve().parents[1] / "backend" / "dist" / "prebenchmark-sidecar" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=default_exe())
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args(argv)

    exe = args.exe.resolve()
    if not exe.is_file():
        print(f"error: sidecar not found: {exe}", file=sys.stderr)
        return 2

    data_dir = Path(tempfile.mkdtemp(prefix="nnb-sidecar-"))
    port = free_port()
    log_path = data_dir / "sidecar.log"
    log_file = log_path.open("w", encoding="utf-8", errors="replace")

    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    process = subprocess.Popen(
        [str(exe), "--data-dir", str(data_dir), "--port", str(port)],
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=flags,
    )

    try:
        print(f"sidecar pid={process.pid} port={port} data={data_dir}")
        wait_health(port, READY_TIMEOUT)
        print("health: ok")

        payload = {
            "dataset": "wine_quality",
            "activation": "relu",
            "optimizer": "adam",
            "lr": 0.001,
            "init": "he",
            "class_weight": "none",
            "seed": 42,
            "max_epochs": 1,
            "early_stop": False,
            "include_test": False,
            "device": "cpu",
        }
        created = request("POST", f"http://127.0.0.1:{port}/api/runs", payload)
        job_id = created["job_id"]
        print(f"job: {job_id} ({created['status']})")

        deadline = time.monotonic() + RUN_TIMEOUT
        snapshot: dict = {}
        while time.monotonic() < deadline:
            snapshot = request("GET", f"http://127.0.0.1:{port}/api/runs/{job_id}")
            if snapshot.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.5)

        state = snapshot.get("execution_state")
        epochs = snapshot.get("epochs")
        print(f"status={snapshot.get('status')} execution_state={state} epochs={epochs}")
        if snapshot.get("status") != "completed":
            raise RuntimeError(f"run did not complete: {snapshot.get('error')}")
        if state not in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
            raise RuntimeError(f"unexpected execution state: {state}")
        if not epochs:
            raise RuntimeError("the run executed zero epochs")

        persisted = data_dir / "results" / "ui-runs" / "ui-exploratory"
        runs = [item for item in persisted.iterdir() if item.is_dir()] if persisted.is_dir() else []
        print(f"persisted runs: {len(runs)}")

        print("SMOKE OK")
        return 0
    except Exception as error:  # noqa: BLE001 - report the failure with context
        print(f"SMOKE FAILED: {error}", file=sys.stderr)
        log_file.flush()
        print("--- sidecar log ---", file=sys.stderr)
        print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        return 1
    finally:
        try:
            if process.stdin:
                process.stdin.write("shutdown\n")
                process.stdin.close()
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        log_file.close()
        if not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
