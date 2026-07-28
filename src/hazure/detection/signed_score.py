"""A signed scorer thresholded on magnitude and filtered on direction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hazure import BaseScorer, BaseThreshold, TimeSeries

__all__ = [
    "SignedScoreDetector",
]


from hazure.detection.score import ScoreDetector
from hazure.detection.side import Side, check_side


class SignedScoreDetector(ScoreDetector):
    """A signed score, thresholded on magnitude and filtered by direction.

    The threshold judges ``|score|``, so "how big is too big" is asked once and
    answered symmetrically. The sign of the score then says which way the series
    moved, and ``side`` decides whether that direction is of interest. Detecting
    only the increases is therefore not a different algorithm, just a filter on
    the same one.

    Parameters
    ----------
    scorer
        A scorer whose sign is meaningful.
    threshold
        The rule applied to the magnitude. A one-sided rule such as
        ``IqrThreshold(factor=(None, 3.0))`` is usual, since a magnitude has no
        interesting lower tail.
    side
        ``"both"``, ``"positive"`` for increases only, or ``"negative"`` for
        decreases only.

    Raises
    ------
    ValueError
        ``side`` is not one of the three directions.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.scoring import DoubleRollingScorer
    >>> from hazure.thresholds import IqrThreshold
    >>> values = np.array([1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]"), values
    ... )
    >>> detector = SignedScoreDetector(
    ...     DoubleRollingScorer(window=(3, 1), diff="diff"),
    ...     IqrThreshold(factor=(None, 3.0)),
    ...     side="positive",
    ... )
    >>> detector.fit_detect(ts).values.ravel()
    array([nan, nan, nan,  1.,  0.,  0.,  0.,  0.])
    """

    side: Side

    def __init__(
        self,
        scorer: BaseScorer | None,
        threshold: BaseThreshold,
        side: Side = "both",
    ) -> None:
        check_side(side)
        self.scorer = scorer
        self.threshold = threshold
        self.side = side
        self._build()

    def _build(self) -> None:
        check_side(self.side)

    def _tested(self, scores: TimeSeries) -> TimeSeries:
        return scores.wrap(np.abs(scores.values), scores.columns)

    def _labels(self, scores: TimeSeries) -> TimeSeries:
        labels = self.threshold.run(self._tested(scores))
        return _gate(labels, scores, self.side)


def _gate(labels: TimeSeries, signed: TimeSeries, side: Side) -> TimeSeries:
    """Clear labels whose score moved in a direction the caller did not ask for."""
    if side == "both":
        return labels
    values = labels.values
    wanted = signed.values > 0.0 if side == "positive" else signed.values < 0.0
    gated = np.where(wanted, values, 0.0)
    # An unknown score stays unknown: withholding a direction is a statement
    # about scores that exist, not about scores that are missing.
    gated[np.isnan(values)] = np.nan
    return labels.wrap(gated, labels.columns)
