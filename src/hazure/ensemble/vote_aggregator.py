"""Anomalous if enough of the inputs that could say anything say so."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseAggregator
from hazure.ensemble.states import _states

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "VoteAggregator",
]


class VoteAggregator(BaseAggregator):
    """Flag a point when enough of the inputs do.

    The middle ground between :class:`OrAggregator` and :class:`AndAggregator`,
    and the natural way to ensemble many detectors of similar quality: one
    detector's false positive is outvoted, while a genuine anomaly that most of
    them see survives.

    Unknown inputs abstain rather than vote against, so a detector that cannot
    label the start of a series does not drag the fraction down there.

    Parameters
    ----------
    threshold
        Fraction of the *known* inputs that must flag a point, from 0 to 1
        inclusive. The default of 0.5 is a simple majority, counting a tie as
        anomalous. At 1 every known input must agree; at 0 every row with any
        known label is flagged.

    Raises
    ------
    ValueError
        ``threshold`` lies outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, np.nan, np.nan]]),
    ...     ["a", "b", "c"],
    ... )
    >>> VoteAggregator(threshold=0.5).aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 1.0]
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        if not 0.0 <= self.threshold <= 1.0:
            msg = (
                f"threshold must be a fraction in [0, 1], got {self.threshold}. "
                f"It is the share of inputs that must agree, not a count."
            )
            raise ValueError(msg)

        anomalous, unknown = _states(ts.values)
        known = (~unknown).sum(axis=1)
        votes = anomalous.sum(axis=1)
        # A row where every input abstains divides zero by zero; the result is
        # discarded below, so the warning it would raise is not worth printing.
        with np.errstate(invalid="ignore", divide="ignore"):
            share = votes / known
        combined = np.where(known > 0, (share >= self.threshold).astype(float), np.nan)
        return ts.wrap(combined, ["anomaly"])
