"""Combining scores rather than verdicts, before anything has been binarised.

Thresholding each input first discards the one thing an ensemble usually wants
to keep: how far each detector thought the point was from normal. This module
combines the scores instead, and its real work is making scores in unrelated
units comparable before it does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from hazure import BaseAggregator
from hazure.thresholds import MAD_SCALE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "ScoreAggregator",
]


#: How :class:`ScoreAggregator` reduces one row of normalised scores.
How = Literal["mean", "max", "median"]


#: How :class:`ScoreAggregator` makes the input columns comparable first.
Normalize = Literal["rank", "robust", "none"]


class ScoreAggregator(BaseAggregator):
    """Combine several anomaly scores into one score.

    The label aggregators reconcile verdicts, which means each input has already
    been reduced to yes or no. That is a lossy step to take first: a point two
    scorers *nearly* flagged becomes indistinguishable from one neither came
    close to flagging, and the ranking — the thing an operator actually works
    down, worst first — is gone. Combining the scores keeps it, which is why
    score-level ensembling is the usual way anomaly-detection ensembles are
    built.

    What makes it harder than combining labels is that scores are not
    commensurable. A reconstruction error in squared units, a robust z-score and
    a rolling-quantile exceedance have no common yardstick, so averaging them raw
    is not an average of opinions but a vote weighted by whichever scorer happens
    to emit the largest numbers. ``normalize`` is the answer to that, and is the
    parameter worth thinking about.

    Parameters
    ----------
    how
        How one row of normalised scores is reduced across the inputs.
        ``"mean"`` asks for consensus: a single input screaming while the rest
        are quiet is diluted, which suppresses one detector's false positives
        and also delays a real anomaly that only one detector was built to see.
        ``"max"`` is the opposite bargain — the loudest input carries the row, so
        nothing any input noticed is lost and nothing any input imagined is
        either. ``"median"`` sits between them: unmoved by one input's excursion
        in either direction, but rising only once at least half the inputs do.
    normalize
        How each input column is put on a comparable scale before combining.
        ``"rank"`` replaces each column's observed values by their rank within
        that column, ties averaged, linearly scaled so the smallest observed
        value is 0 and the largest is 1. It is distribution-free and bounded,
        so a scorer cannot buy influence with its units or with one enormous
        outlier, and it is the default because "which points does this scorer
        think are the worst" is all an ensemble needs from a member.
        ``"robust"`` divides each column's distance from its own median by
        :data:`hazure.thresholds.MAD_SCALE` times its median absolute deviation.
        It keeps the *spacing* between scores that ranking flattens away — a
        score ten deviations out stays ten times as alarming as one deviation
        out — at the price of being unbounded again, so one input can still
        dominate a ``"mean"``. ``"none"`` combines the raw scores, and is
        correct only when the inputs already share a scale: two residuals of the
        same series, or several columns out of one scorer.

    Raises
    ------
    ValueError
        ``how`` or ``normalize`` is not one of the listed choices.

    Notes
    -----
    Nothing is learned, so :attr:`trainable` is False and there is no ``fit``.
    The normalisation is recomputed from the very series being combined, as
    :class:`hazure.StandardScale` does, rather than fitted on a training period.
    For ``"rank"`` that is not a shortcut but the definition: a rank exists only
    relative to a sample, and there is no sample here but the one in hand.
    ``"robust"`` could honestly be fitted, and if a fixed yardstick is what you
    want, put a :class:`hazure.DeviationScorer` in front of each input and
    combine with ``normalize="none"``.

    ``NaN`` means unknown and abstains rather than propagating, exactly as it
    does in :class:`VoteAggregator`: a row with two known scores and one unknown
    is the combination of the two, because a detector still inside its warm-up
    window has no opinion to contribute and should not be read as a quiet one.
    A row where every input is unknown is ``NaN``.

    A column whose median absolute deviation is zero has no observed spread to
    divide by — a mostly constant score with a few excursions is enough to do
    that. Under ``"robust"`` such a column is centred and left unscaled, which
    is what :class:`hazure.StandardScale` does with a constant series: values on
    the median contribute exactly 0.0, and the excursions stay finite and
    ordered instead of becoming an infinity that would swallow every other
    input's contribution to a mean.

    Examples
    --------
    Two scorers whose scores differ by three orders of magnitude. Ranking makes
    the units irrelevant, so the expensive scorer's larger numbers do not decide
    the outcome on their own:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([[0.1, 300.0], [0.2, 100.0], [0.9, 200.0], [0.3, np.nan]]),
    ...     ["cheap", "expensive"],
    ... )
    >>> averaged = ScoreAggregator().aggregate(ts)["anomaly"]
    >>> [round(float(value), 3) for value in averaged]
    [0.5, 0.167, 0.75, 0.667]

    The last row's second input is unknown, so it is the first input alone.
    Taking the maximum instead lets whichever input is loudest carry the row:

    >>> loudest = ScoreAggregator(how="max").aggregate(ts)["anomaly"]
    >>> [round(float(value), 3) for value in loudest]
    [1.0, 0.333, 1.0, 0.667]
    """

    trainable: ClassVar[bool] = False

    def __init__(self, how: How = "mean", normalize: Normalize = "rank") -> None:
        _check_choice(how, ("mean", "max", "median"), "how")
        _check_choice(normalize, ("rank", "robust", "none"), "normalize")
        self.how = how
        self.normalize = normalize

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        # Re-checked here as well as in __init__ because set_params() assigns
        # attributes directly, and a typo should fail loudly rather than fall
        # through to a silent default.
        _check_choice(self.how, ("mean", "max", "median"), "how")
        _check_choice(self.normalize, ("rank", "robust", "none"), "normalize")

        scores = ts.values
        if self.normalize == "rank":
            prepared = np.column_stack([_ranks(column) for column in scores.T])
        elif self.normalize == "robust":
            prepared = np.column_stack([_robust(column) for column in scores.T])
        else:
            prepared = scores

        combined = np.full(ts.n_rows, np.nan)
        # Rows with nothing known are left as NaN and excluded from the reduction
        # rather than passed to it, which keeps numpy from warning about the
        # all-NaN slices it would otherwise have to return NaN for anyway.
        known = ~np.isnan(prepared).all(axis=1)
        if known.any():
            rows = prepared[known]
            if self.how == "mean":
                combined[known] = np.nanmean(rows, axis=1)
            elif self.how == "max":
                combined[known] = np.nanmax(rows, axis=1)
            else:
                combined[known] = np.nanmedian(rows, axis=1)
        return ts.wrap(combined, ["anomaly"])


def _ranks(column: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank a column's observed values, ties averaged, scaled onto ``[0, 1]``."""
    observed = ~np.isnan(column)
    ranked = np.full(column.shape, np.nan)
    values = column[observed]
    size = values.size
    if size == 0:
        return ranked
    if size == 1:
        # One observation has nothing to be ranked against. 0.5 declines to call
        # it either the best or the worst point, and is the limit of the constant
        # column below, where every averaged rank lands in the middle.
        ranked[observed] = 0.5
        return ranked

    order = np.argsort(values, kind="stable")
    ascending = values[order]
    # Start of each run of equal values, and one past its end. Positions within a
    # run are the consecutive integers start+1 .. end, whose mean is the midrank
    # that ties share.
    starts = np.flatnonzero(np.r_[True, ascending[1:] != ascending[:-1]])
    ends = np.r_[starts[1:], size]
    midranks = (starts + ends + 1) / 2.0

    positions = np.empty(size, dtype=np.float64)
    positions[order] = np.repeat(midranks, ends - starts)
    # A constant column gives every value the same midrank, (size + 1) / 2, hence
    # 0.5 everywhere: no ranking information, and no input either.
    ranked[observed] = (positions - 1.0) / (size - 1.0)
    return ranked


def _robust(column: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centre a column on its median and scale it by its MAD."""
    values = column[~np.isnan(column)]
    if values.size == 0:
        return column.copy()
    centre = float(np.median(values))
    scale = MAD_SCALE * float(np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale == 0.0:
        # No usable spread, so there is no unit to divide by; leave the column
        # centred but unscaled, as StandardScale does with a constant series.
        scale = 1.0
    return (column - centre) / scale


def _check_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    """Reject a parameter that is not one of a small set of names."""
    if value not in allowed:
        msg = f"{name}={value!r} is not one of {list(allowed)}."
        raise ValueError(msg)
