"""Reading the parameter specs the window transformers accept."""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _pair(spec: _T | tuple[_T, _T]) -> tuple[_T, _T]:
    """Expand a single value into a pair, or pass a 2-tuple through."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            msg = (
                f"Expected one value or a (left, right) 2-tuple, got {len(spec)} items."
            )
            raise ValueError(msg)
        return spec
    return spec, spec


def _is_sequence(value: Any) -> bool:
    """Report whether a parameter asks for several values rather than one."""
    return isinstance(value, (list, tuple, np.ndarray))
