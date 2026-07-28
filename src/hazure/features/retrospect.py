"""Lagging a series onto itself, one column per lag.

Point a regression at the result and it becomes an autoregressive model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseTransformer

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "Retrospect",
]


# ---------------------------------------------------------------------------
# lagging and scaling
# ---------------------------------------------------------------------------


class Retrospect(BaseTransformer):
    """Emit lagged copies of the series as columns.

    The result is the design matrix for autoregression: row ``t`` holds the
    values at ``t - till``, ``t - till - step_size``, and so on, which is what a
    model needs to learn how a control's effect is delayed and how long it
    lasts. Columns are named ``t-0``, ``t-1``, ...; a negative lag looks ahead
    and is named ``t+1``, ``t+2``, ...

    Parameters
    ----------
    n_steps
        Number of lagged columns.
    step_size
        Gap in observations between consecutive columns.
    till
        Nearest lag, in observations. 0 is the current point.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([0.0, 1.0, 2.0, 3.0]),
    ... )
    >>> lagged = Retrospect(n_steps=2, step_size=2, till=1).run(ts)
    >>> lagged.columns
    ('t-1', 't-3')
    >>> lagged.values
    array([[nan, nan],
           [ 0., nan],
           [ 1., nan],
           [ 2.,  0.]])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, n_steps: int = 1, step_size: int = 1, till: int = 0) -> None:
        self.n_steps = n_steps
        self.step_size = step_size
        self.till = till

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if self.n_steps < 1:
            msg = f"n_steps must be at least 1, got {self.n_steps}."
            raise ValueError(msg)
        if self.step_size < 1:
            msg = (
                f"step_size must be at least 1, got {self.step_size}; a step of "
                f"0 would emit the same lag several times."
            )
            raise ValueError(msg)
        if ts.freq is None:
            msg = (
                "Retrospect needs a regular time axis: shifting by a number of "
                "observations means nothing when the sampling interval varies. "
                "Resample the series first, or use RollingAggregate with a "
                "duration window."
            )
            raise ValueError(msg)

        values = ts.values[:, 0]
        lags = self.till + np.arange(self.n_steps) * self.step_size
        lagged = np.full((ts.n_rows, self.n_steps), np.nan, dtype=np.float64)
        # One pass per column, which is n_steps iterations rather than n_rows.
        for position, lag in enumerate(int(value) for value in lags):
            if abs(lag) >= ts.n_rows:
                continue
            if lag >= 0:
                lagged[lag:, position] = values[: ts.n_rows - lag]
            else:
                lagged[: ts.n_rows + lag, position] = values[-lag:]
        return ts.wrap(lagged, [_lag_name(int(lag)) for lag in lags])


def _lag_name(lag: int) -> str:
    """Name a column after its lag, keeping the sign readable."""
    return f"t-{lag}" if lag >= 0 else f"t+{-lag}"
