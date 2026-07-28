"""A rolling statistic asserted to be the score itself."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hazure.features import (
    RollingAggregate,
)
from hazure.scoring.transformer_scorer import TransformerScorer

if TYPE_CHECKING:
    from hazure import BaseTransformer
    from hazure._core.window import Closed, Window

__all__ = [
    "RollingAggregateScorer",
]


class RollingAggregateScorer(TransformerScorer):
    """Score each point by a statistic of the window ending at it.

    A feature-style score: not anomalous or not on its own, but a summary the
    threshold can be pointed at. Counting non-zero values in a rolling day, for
    instance, turns "the pump idled" into a number a quantile rule can judge.

    Parameters
    ----------
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``).
    agg
        A name from :data:`hazure.AGGREGATIONS`.
    center
        Centre the window on each point instead of trailing it.
    min_periods
        Minimum non-missing observations for a result. Defaults to the full
        window for integer windows and to 1 for duration windows.
    closed
        Which window endpoints to include; defaults to ``"right"``.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 1.0, 1.0, 9.0, 1.0, 1.0])
    >>> RollingAggregateScorer(window=2, agg="max").score(ts).values.ravel()
    array([nan,  1.,  1.,  9.,  9.,  1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window,
        agg: str = "mean",
        center: bool = False,
        min_periods: int | None = None,
        closed: Closed | None = None,
        q: float | None = None,
    ) -> None:
        self.window = window
        self.agg = agg
        self.center = center
        self.min_periods = min_periods
        self.closed = closed
        self.q = q

    def _new_transformer(self) -> BaseTransformer:
        return RollingAggregate(
            window=self.window,
            agg=self.agg,
            agg_params=None if self.q is None else {"q": self.q},
            center=self.center,
            min_periods=self.min_periods,
            closed=self.closed,
        )
