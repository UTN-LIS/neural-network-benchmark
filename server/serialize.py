"""Conversión de valores del dominio a JSON estricto.

El runner produce ``float``/``int`` nativos en las filas de época, pero los
metadatos incluyen objetos de NumPy (por ejemplo ``np.float64`` y arrays) y
``Path``. ``json.dumps`` no acepta esos tipos, y JSON estricto tampoco acepta
``NaN``/``Infinity``: ambos se convierten a ``null``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Devuelve una estructura compuesta solo por tipos JSON estrictos."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:  # noqa: BLE001 - conversión de mejor esfuerzo
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # noqa: BLE001 - conversión de mejor esfuerzo
            pass

    return str(value)
