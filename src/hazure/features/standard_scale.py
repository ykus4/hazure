"""Putting a series on a standard scale.

Recomputed from whatever series it is handed rather than learned, which is
what separates it from :class:`hazure.scoring.DeviationScorer`: this is for
making columns comparable, not for measuring departures from a fitted
normal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseTransformer

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "StandardScale",
]


class StandardScale(BaseTransformer):
    """Centre and scale a series by its own mean and standard deviation.

    Scaling is per-series and computed from the series being transformed, so a
    frame of columns in different units becomes comparable without training.
    The standard deviation is the sample one (``ddof=1``); a constant series has
    none, and is centred but left unscaled rather than divided by zero.
    """

    trainable: ClassVar[bool] = False

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        values = ts.values[:, 0]
        present = values[~np.isnan(values)]
        if present.size == 0:
            return ts.wrap(values)
        spread = float(present.std(ddof=1)) if present.size > 1 else 0.0
        if not np.isfinite(spread) or spread == 0.0:
            spread = 1.0
        return ts.wrap((values - float(present.mean())) / spread)
