"""Pairing a scorer with a threshold.

A detector is the common case made convenient: score the series, then draw the
line, in one object with one set of parameters. The two halves stay visible as
:attr:`ScoreDetector.scorer` and :attr:`ScoreDetector.threshold`, so a detector
can be taken apart — to look at the raw score, to reuse the scorer with a
different rule, or to see exactly what a named detector is made of.

Two variations cover every detector in hazure:

* whether the algorithm needs one column at a time or all of them at once, which
  decides whether a frame fans out into independent per-column models;
* whether the score is *signed*, in which case the threshold judges its magnitude
  and its sign decides whether the direction is one the caller asked about.

That second point is what ``side`` means, and it is implemented once here rather
than in each detector that offers it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, Literal, get_args

import numpy as np

from hazure import BaseDetector

if TYPE_CHECKING:
    from hazure import BaseScorer, BaseThreshold, TimeSeries

__all__ = [
    "MultivariateScoreDetector",
    "MultivariateSignedScoreDetector",
    "ScoreDetector",
    "Side",
    "SignedScoreDetector",
]

#: Which direction of excursion counts as an anomaly.
Side = Literal["both", "positive", "negative"]

_SIDES: Final = get_args(Side)

#: Column name every multivariate detector reports its labels under.
_LABEL_NAME: Final = "anomaly"


def check_side(side: object) -> None:
    """Reject a ``side`` that is not one of the three directions.

    Parameters
    ----------
    side
        The value to check.

    Raises
    ------
    ValueError
        ``side`` is not ``"both"``, ``"positive"`` or ``"negative"``.
    """
    if side not in _SIDES:
        msg = f"side={side!r} is not one of {list(_SIDES)}."
        raise ValueError(msg)


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


def _as_frame_label(labels: TimeSeries) -> TimeSeries:
    """Rename a whole-frame verdict, which belongs to no single column."""
    return labels.wrap(labels.values, [_LABEL_NAME])


class MultivariateScoreDetector(ScoreDetector):
    """A pairing whose scorer needs every column at once.

    Fitting sees the whole frame rather than one column at a time, so the model
    can learn how the columns relate. The single label series is reported under
    the column name ``anomaly``, since it describes the frame as a whole and not
    any one of its columns.
    """

    multivariate: ClassVar[bool] = True

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return _as_frame_label(super()._compute(ts))


class MultivariateSignedScoreDetector(SignedScoreDetector):
    """A signed pairing whose scorer needs every column at once.

    Combines the direction filter of :class:`SignedScoreDetector` with the
    whole-frame view of :class:`MultivariateScoreDetector`.
    """

    multivariate: ClassVar[bool] = True

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return _as_frame_label(super()._compute(ts))
