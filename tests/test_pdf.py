"""Tests for the HTML -> PDF helper (``prebenchmark.pdf``).

No browser is launched: ``subprocess.run`` is stubbed so the tests stay offline
and deterministic on every machine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from prebenchmark import pdf


def _html(tmp_path: Path) -> Path:
    path = tmp_path / "report.html"
    path.write_text("<html><body>report</body></html>", encoding="utf-8")
    return path


def test_missing_html_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no se encontró el HTML"):
        pdf.html_to_pdf(tmp_path / "missing.html", tmp_path / "out.pdf")


def test_without_browser_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pdf, "find_browser", lambda: None)
    with pytest.raises(RuntimeError, match="no se encontró Chrome ni Edge"):
        pdf.html_to_pdf(_html(tmp_path), tmp_path / "out.pdf")


def test_browser_is_invoked_with_print_to_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = _html(tmp_path)
    browser = tmp_path / "chrome.exe"
    browser.write_text("", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        target = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--print-to-pdf="))
        Path(target).write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(pdf.subprocess, "run", fake_run)
    output = pdf.html_to_pdf(html, tmp_path / "out.pdf", browser=browser)

    assert output.is_file()
    assert output.read_bytes().startswith(b"%PDF")
    assert "--headless" in captured["command"]
    assert any(arg.startswith("--print-to-pdf=") for arg in captured["command"])
    assert html.as_uri() in captured["command"]


def test_browser_without_output_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = _html(tmp_path)
    browser = tmp_path / "chrome.exe"
    browser.write_text("", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "boom")

    monkeypatch.setattr(pdf.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no generó el PDF"):
        pdf.html_to_pdf(html, tmp_path / "out.pdf", browser=browser)


def test_browser_candidates_only_returns_existing_files():
    for candidate in pdf.browser_candidates():
        assert candidate.is_file()
