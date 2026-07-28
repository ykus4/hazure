"""A scorer and a threshold, fitted together."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure import BaseDetector

if TYPE_CHECKING:
    from hazure import BaseScorer, BaseThreshold, TimeSeries

__all__ = [
    "ScoreDetector",
]


class ScoreDetector(BaseDetector):
    """A scorer and a threshold, applied in that order.

    Fitting fits the scorer, then fits the threshold on the scores the fitted
    scorer produces for the training data — so the threshold learns the scale the
    scorer actually works on, which is the whole reason the two are fitted
    together rather than independently.

    Parameters
    ----------
    scorer
        The scorer to apply, or None when the series is already its own score,
        as it is for a plain value range.
    threshold
        The rule that turns those scores into labels.

    Examples
    --------
    Any scorer pairs with any threshold:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.scoring import DeviationScorer
    >>> from hazure.thresholds import MadThreshold
    >>> values = np.array([5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 40.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]"), values
    ... )
    >>> detector = ScoreDetector(DeviationScorer(), MadThreshold())
    >>> detector.fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 1.])
    >>> detector.scorer.center_
    6.0
    """

    scorer: BaseScorer | None
    threshold: BaseThreshold

    def __init__(self, scorer: BaseScorer | None, threshold: BaseThreshold) -> None:
        self.scorer = scorer
        self.threshold = threshold
        self._build()

    def _build(self) -> None:
        """Configure the composed parts from this detector's parameters.

        Called at construction and again at every :meth:`fit`, so a parameter
        changed with ``set_params`` takes effect on the next fit. The generic
        pairing is handed its parts directly and so has nothing to do; the named
        detectors build theirs here.
        """

    def _learn(self, ts: TimeSeries) -> None:
        self._build()
        scores = ts if self.scorer is None else self.scorer.fit(ts).run(ts)
        self.threshold.fit(self._tested(scores))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = ts if self.scorer is None else self.scorer.run(ts)
        return self._labels(scores)

    def _tested(self, scores: TimeSeries) -> TimeSeries:
        """Return the series the threshold is applied to."""
        return scores

    def _labels(self, scores: TimeSeries) -> TimeSeries:
        """Turn scores into labels."""
        return self.threshold.run(self._tested(scores))
