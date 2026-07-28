"""Shared machinery for the fence-shaped rules.

Every rule but the ESD test reduces to a pair of cut-offs and one
comparison: strictly outside is anomalous, a NaN score keeps a NaN label,
and an unknown fence makes every label unknown.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "MAD_SCALE",
    "Factor",
    "FactorSpec",
]


#: Scale factor that turns a median absolute deviation into an estimate of the
#: standard deviation of a normal sample. For X ~ N(mu, sigma) the median of
#: |X - mu| is sigma * Phi^-1(0.75) = 0.6745 * sigma, so dividing the MAD by that
#: constant — equivalently multiplying by 1 / 0.6745 = 1.4826 — puts the MAD on
#: the same scale as a standard deviation. Without it, ``factor=3`` would mean
#: three MADs, which is only two standard deviations.
MAD_SCALE: Final = 1.482602218505602


#: A tail factor: a number, or None to leave that tail unbounded.
Factor = float | None


#: One factor for both tails, or ``(low, high)``.
FactorSpec = Factor | tuple[Factor, Factor]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _label(values: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """Flag values outside ``[low, high]``, leaving missing scores unknown."""
    if math.isnan(low) or math.isnan(high):
        # The cut-offs were never learned from valid data, so no label can be
        # justified for any point.
        return np.full(values.shape, np.nan, dtype=np.float64)
    labels = ((values > high) | (values < low)).astype(np.float64)
    labels[np.isnan(values)] = np.nan
    return labels


def _valid(ts: TimeSeries) -> NDArray[np.float64]:
    """Return the non-missing values of a univariate series."""
    column = ts.values[:, 0]
    return column[~np.isnan(column)]


def _factors(spec: FactorSpec, name: str) -> tuple[Factor, Factor]:
    """Expand a scalar-or-pair factor into ``(low, high)``.

    Parameters
    ----------
    spec
        One factor for both tails, or ``(low, high)``. ``None`` on a side means
        that side is unbounded.
    name
        Parameter name, for error messages.

    Returns
    -------
    tuple
        The low-side and high-side factors.

    Raises
    ------
    ValueError
        The pair is not of length two, or a factor is negative.
    """
    if isinstance(spec, tuple):
        pair: tuple[Factor, ...] = spec
        if len(pair) != 2:
            msg = (
                f"{name} must be a number, None, or a (low, high) pair; got "
                f"{len(pair)} items."
            )
            raise ValueError(msg)
        low, high = pair
    else:
        low, high = spec, spec

    for side, value in (("low", low), ("high", high)):
        if value is not None and value < 0:
            msg = (
                f"The {side} side of {name}={spec!r} is negative, which would "
                f"invert the normal range. Use a non-negative factor, or None "
                f"to leave that side unbounded."
            )
            raise ValueError(msg)
    return low, high


def _bound(centre: float, spread: float, factor: Factor, *, upper: bool) -> float:
    """Offset ``centre`` by ``factor * spread``, or go unbounded for ``None``."""
    if factor is None:
        return math.inf if upper else -math.inf
    return centre + factor * spread if upper else centre - factor * spread


def _require_a_bound(low: object, high: object, name: str) -> None:
    """Reject a threshold that could never flag anything."""
    if low is None and high is None:
        msg = (
            f"{name} needs at least one of low=... or high=...; with both unset "
            f"it can never flag a point."
        )
        raise ValueError(msg)
