"""Flagging the breakpoints PELT finds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.pelt_scorer import PeltScorer
from hazure.thresholds import FixedThreshold

if TYPE_CHECKING:
    from hazure.methods.breakpoint_scorer import Cost


__all__ = [
    "PeltDetector",
]


class PeltDetector(ScoreDetector):
    """Flag the points at which the series changed regime.

    :class:`PeltScorer` with a threshold that passes its non-zero scores through.
    There is deliberately no factor to tune: the penalty has already decided which
    changes are large enough to be worth a segment, and second-guessing that with
    a rule on the score would be answering the same question twice with less
    information. To report fewer changes, raise ``penalty``.

    Parameters
    ----------
    penalty
        Cost of admitting one more segment. None derives a BIC-style value from
        the data.
    cost
        ``"l2"`` for squared deviations from the segment mean, ``"l1"`` for
        absolute deviations from its median.
    min_size
        Shortest segment allowed, in observations.
    jump
        Consider only breakpoints at multiples of this many observations.

    Raises
    ------
    ValueError
        ``cost`` is unknown, or a size is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> values = np.concatenate([rng.normal(size=60), rng.normal(loc=10.0, size=60)])
    >>> time = np.arange("2024-01-01", "2024-04-30", dtype="datetime64[D]")
    >>> labels = PeltDetector().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([60])
    """

    def __init__(
        self,
        penalty: float | None = None,
        cost: Cost = "l2",
        min_size: int = 2,
        jump: int = 1,
    ) -> None:
        self.penalty = penalty
        self.cost = cost
        self.min_size = min_size
        self.jump = jump
        self._build()

    def _build(self) -> None:
        self.scorer = PeltScorer(
            penalty=self.penalty,
            cost=self.cost,
            min_size=self.min_size,
            jump=self.jump,
        )
        # Every change the segmentation kept is a change worth reporting, so the
        # line goes at zero.
        self.threshold = FixedThreshold(high=0.0)
