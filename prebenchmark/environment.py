"""Captura de mejor esfuerzo del entorno de ejecución.

Cada ejecución registra *dónde* se ejecutó (hardware) y *con qué* (versiones de bibliotecas),
de modo que un resultado pueda razonarse y reproducirse más tarde. Nada de esto es necesario
para el entrenamiento: cada sonda está envuelta de forma defensiva y degrada a ``None`` cuando un
valor no está disponible, por lo que capturar el entorno nunca hace fallar una ejecución.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

__all__ = ["capture_environment", "capture_hardware", "capture_library_versions"]


def _cpu_description() -> str | None:
    processor = platform.processor()
    if processor:
        return processor
    machine = platform.machine()
    return machine or None


def capture_hardware() -> dict[str, Any]:
    """Devuelve una descripción de mejor esfuerzo de la máquina que ejecutó el experimento."""
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": _cpu_description(),
        "cpu_count": _safe_cpu_count(),
        "python_version": platform.python_version(),
    }
    try:
        import torch
    except Exception:  # pragma: no cover - torch es una dependencia obligatoria, solo defensivo
        hardware["cuda_available"] = None
        hardware["device_count"] = 0
        hardware["devices"] = []
        return hardware

    try:
        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = [torch.cuda.get_device_name(i) for i in range(count)] if cuda_available else []
    except Exception:  # pragma: no cover - defensivo: nunca fallar en la sonda
        cuda_available, count, devices = False, 0, []
    hardware["cuda_available"] = cuda_available
    hardware["device_count"] = count
    hardware["devices"] = devices
    return hardware


def _safe_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:  # pragma: no cover - defensivo
        return None


def capture_library_versions() -> dict[str, Any]:
    """Devuelve las versiones de Python y de las bibliotecas de las que depende el pipeline."""
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
    }
    for name in ("torch", "numpy", "pandas", "sklearn", "scipy", "matplotlib"):
        versions[name] = _module_version(name)
    return versions


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:  # pragma: no cover - dependencia opcional/ausente
        return None
    return getattr(module, "__version__", None)


def capture_environment() -> dict[str, Any]:
    """Devuelve ``{"hardware": ..., "library_versions": ...}`` para un registro de ejecución."""
    return {
        "hardware": capture_hardware(),
        "library_versions": capture_library_versions(),
    }
