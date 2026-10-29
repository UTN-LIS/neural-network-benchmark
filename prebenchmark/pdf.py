"""Generación de PDF a partir del reporte HTML usando Chrome o Edge headless.

El reporte HTML es autocontenido, así que imprimirlo a PDF no requiere servidor,
red ni dependencias de Python: alcanza con un navegador Chromium instalado. El
módulo busca el navegador en las ubicaciones habituales de Windows, macOS y Linux
y ejecuta el modo ``--print-to-pdf``.

Si no hay navegador disponible, el HTML sigue siendo válido; el llamador decide
si eso es un error fatal o una advertencia.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

__all__ = ["browser_candidates", "find_browser", "html_to_pdf"]

_WINDOWS_CANDIDATES: tuple[str, ...] = (
    r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
    r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
    r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
    r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
)

_MACOS_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

_LINUX_COMMANDS: tuple[str, ...] = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
)


def browser_candidates() -> list[Path]:
    """Devuelve las rutas de navegador candidatas que existen en este sistema."""
    import os

    candidates: list[Path] = []
    for template in _WINDOWS_CANDIDATES:
        expanded = os.path.expandvars(template)
        if "%" not in expanded:
            candidates.append(Path(expanded))
    for template in _MACOS_CANDIDATES:
        candidates.append(Path(template))
    for command in _LINUX_COMMANDS:
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))
    return [candidate for candidate in candidates if candidate.is_file()]


def find_browser() -> Path | None:
    """Primer navegador Chromium disponible, o ``None``."""
    candidates = browser_candidates()
    return candidates[0] if candidates else None


def html_to_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    *,
    browser: str | Path | None = None,
    timeout: int = 180,
) -> Path:
    """Imprime ``html_path`` a ``pdf_path`` con Chrome/Edge headless.

    Lanza ``FileNotFoundError`` si el HTML no existe y ``RuntimeError`` si no hay
    navegador o si la impresión no produce el PDF.
    """
    html = Path(html_path).resolve()
    pdf = Path(pdf_path).resolve()
    if not html.is_file():
        raise FileNotFoundError(f"no se encontró el HTML: {html}")

    browser_path = Path(browser).expanduser() if browser is not None else find_browser()
    if browser_path is None or not browser_path.is_file():
        raise RuntimeError(
            "no se encontró Chrome ni Edge; instalá uno o pasá la ruta con --browser"
        )

    pdf.parent.mkdir(parents=True, exist_ok=True)
    command: Sequence[str] = (
        str(browser_path),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf}",
        html.as_uri(),
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"la impresión a PDF excedió {timeout} s") from error

    if not pdf.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[-1]
        raise RuntimeError(f"el navegador no generó el PDF: {detail or 'sin detalle'}")
    return pdf
