"""Scoring a score: how well the ranking separates anomalies, with no threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hazure.evaluation.metrics import _binary, _dispatch, _joined

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure.events import Events

__all__ = [
    "average_precision",
    "roc_auc",
]


def average_precision(y_true: Any, scores: Any) -> float | dict[str, float]:
    """Area under the precision-recall curve of a continuous score.

    Every other metric in this module needs labels, so evaluating a *scorer* with
    them means picking a threshold first and then measuring the threshold as much
    as the score. This does not: it asks only whether the anomalous samples are
    ranked above the normal ones, over every threshold at once.

    Parameters
    ----------
    y_true
        Ground truth labels: a series or frame that
        :meth:`hazure.TimeSeries.from_any` accepts, or a dict of those. As
        everywhere else, a label counts as anomalous when it clips to exactly 1,
        and ``NaN`` counts as normal.
    scores
        A continuous score on the same time axis, higher meaning more anomalous —
        the output of any :class:`~hazure.BaseScorer`. Rows whose score is
        ``NaN`` are dropped: an unknown score cannot be ranked.

    Returns
    -------
    float or dict of float
        Average precision in ``[0, 1]``, or one per column / dict key. ``nan``
        when the answer is undefined: no positives, no negatives, or nothing left
        once the ``NaN`` scores are dropped.

    Raises
    ------
    TypeError
        Either argument is an ``Events`` or a list of intervals, or the two are
        not the same kind of object.
    ValueError
        The two series sit on different time axes, or they carry different column
        names or keys.

    See Also
    --------
    roc_auc : The other threshold-free summary, less sensitive to rarity.

    Notes
    -----
    Sample-based, never event-based. Take the **distinct** score values in
    decreasing order as thresholds. Writing ``tp_k`` and ``fp_k`` for the
    anomalous and normal samples scoring at least as high as the *k*-th of them,
    ``n_P`` for the number of anomalous samples, and summing over the thresholds::

        P_k = tp_k / (tp_k + fp_k)
        R_k = tp_k / n_P,           R_0 = 0
        AP  = sum_k (R_k - R_{k-1}) * P_k

    This is what ``sklearn.metrics.average_precision_score`` computes, and the
    tests check that it agrees. Two details in it are the ones that matter.
    *Distinct* values, so a block of tied scores contributes exactly one point
    and no arbitrary order within the block can flatter the result. And a
    right-hand rectangle rather than a trapezoid, so the curve is never
    interpolated: the segment between two achievable operating points is
    generally not itself achievable, and integrating it would be optimistic.

    There is no event-based counterpart, deliberately. An event-based score would
    need one number per interval, and there is no defensible way to choose it:
    the maximum over the interval rewards a single lucky sample, the mean punishes
    a detector that is right for one minute of a six-hour outage, and either
    choice changes the ranking. Reduce the score to labels and use
    :func:`~hazure.evaluation.recall` and :func:`~hazure.evaluation.precision` if
    events are what you care about.

    Examples
    --------
    A score that separates the two classes cleanly:

    >>> import pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=4, freq="h")
    >>> truth = pd.Series([0, 0, 1, 1], index=index)
    >>> average_precision(truth, pd.Series([0.1, 0.2, 0.8, 0.9], index=index))
    1.0

    A score that says nothing, because every sample ties. Half the samples are
    anomalous, and half is what a coin gets:

    >>> average_precision(truth, pd.Series([0.5] * 4, index=index))
    0.5

    The exactly wrong ranking is not 0.0: the last threshold still has to include
    every sample, and by then precision is the base rate.

    >>> average_precision(truth, pd.Series([0.9, 0.8, 0.2, 0.1], index=index))
    0.41666666666666663
    """
    return _dispatch(
        y_true,
        scores,
        _average_precision,
        _no_events,
        align=_scored,
        guess_name="scores",
    )


def roc_auc(y_true: Any, scores: Any) -> float | dict[str, float]:
    """Area under the ROC curve of a continuous score.

    The probability that a randomly chosen anomalous sample outranks a randomly
    chosen normal one, with ties counting half. It is threshold-free like
    :func:`average_precision`, and differs in what it is sensitive to: because
    the false positive rate has the whole normal class in its denominator, a
    detector can look excellent here while alerting far more often than it is
    right. On the rare-anomaly data this library is aimed at, quote both.

    Parameters
    ----------
    y_true
        Ground truth labels, in any form :func:`average_precision` accepts.
    scores
        A continuous score on the same time axis, higher meaning more anomalous.
        Rows whose score is ``NaN`` are dropped.

    Returns
    -------
    float or dict of float
        Area in ``[0, 1]``, or one per column / dict key. ``nan`` when the answer
        is undefined: no positives, no negatives, or nothing left once the
        ``NaN`` scores are dropped.

    Raises
    ------
    TypeError
        Either argument is an ``Events`` or a list of intervals, or the two are
        not the same kind of object.
    ValueError
        The two series sit on different time axes, or they carry different column
        names or keys.

    See Also
    --------
    average_precision : The other threshold-free summary, harder to flatter.

    Notes
    -----
    Sample-based, never event-based, for the reason given in
    :func:`average_precision`.

    Computed exactly, through midranks rather than by walking a curve. Rank the
    retained samples by score in increasing order from 1, giving every member of
    a tied block the average ``r_i`` of the ranks that block occupies. With ``P``
    the set of anomalous samples, ``n_P`` its size and ``n_N`` the number of
    normal samples, this is the Mann-Whitney statistic::

        AUC = (sum_{i in P} r_i - n_P * (n_P + 1) / 2) / (n_P * n_N)

    The midranks are what handle ties, and they handle them the way the curve
    does: a tied block is one diagonal step of the ROC, so what it contributes is
    the trapezoid under that diagonal, which is half of its tied pairs. Sorting
    the block arbitrarily instead would turn that diagonal into a staircase, and
    a score carrying no information at all could then come out anywhere between 0
    and 1. The tests check the result against
    ``sklearn.metrics.roc_auc_score``, ties included.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=4, freq="h")
    >>> truth = pd.Series([0, 0, 1, 1], index=index)
    >>> roc_auc(truth, pd.Series([0.1, 0.2, 0.8, 0.9], index=index))
    1.0
    >>> roc_auc(truth, pd.Series([0.9, 0.8, 0.2, 0.1], index=index))
    0.0

    Every sample tied is exactly 0.5, whatever order the sort happened to put
    them in:

    >>> roc_auc(truth, pd.Series([0.5] * 4, index=index))
    0.5

    One anomaly ranked top, the other tied with a normal sample halfway down:

    >>> roc_auc(truth, pd.Series([0.5, 0.1, 0.5, 0.9], index=index))
    0.875
    """
    return _dispatch(
        y_true, scores, _roc_auc, _no_events, align=_scored, guess_name="scores"
    )


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _scored(
    truth: TimeSeries, scores: TimeSeries
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Read a label column and a score column off one shared time axis.

    Only the truth becomes boolean. The score is the whole point of these two
    metrics and passes through untouched, ``NaN`` included, since dropping the
    unrankable rows is the kernel's job.
    """
    labels, values = _joined(truth, scores)
    return _binary(labels), values


def _no_events(truth: Events, scores: Events) -> float:
    """Refuse event-based input, which cannot carry a score at all."""
    msg = (
        "average_precision and roc_auc need a continuous score per sample, so "
        "they are sample-based only. Pass a label series and a score series; "
        "an Events has no score to rank."
    )
    raise TypeError(msg)


def _rankable(
    truth: NDArray[np.bool_], scores: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.float64]] | None:
    """Drop unscored rows, or return ``None`` if nothing can be ranked.

    A ``NaN`` score is not a low score: the sample was never placed, most often
    because a rolling window had not filled yet, and placing it at the bottom
    would credit the scorer for a judgement it did not make.
    """
    keep = ~np.isnan(scores)
    truth, scores = truth[keep], scores[keep]
    positives = int(np.count_nonzero(truth))
    # Both classes have to be present: with one class there is no pair to order,
    # and neither area is a ratio of anything.
    if positives == 0 or positives == truth.size:
        return None
    return truth, scores


def _average_precision(truth: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    """Recall-weighted mean precision over the distinct score thresholds."""
    rankable = _rankable(truth, scores)
    if rankable is None:
        return float("nan")
    truth, scores = rankable

    order = np.argsort(-scores, kind="stable")
    descending = scores[order]
    ranked = truth[order]
    hits = np.cumsum(ranked)
    flagged = np.arange(1, ranked.size + 1)

    # One point per distinct score, taken at the end of each tied block: within a
    # block the order is arbitrary, so no threshold can cut through it.
    last = np.append(np.flatnonzero(descending[1:] != descending[:-1]), ranked.size - 1)
    precision = hits[last] / flagged[last]
    recalled = hits[last] / hits[-1]
    return float(np.sum(np.diff(recalled, prepend=0.0) * precision))


def _roc_auc(truth: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    """Mann-Whitney statistic over midranks, which is the tie-aware ROC area."""
    rankable = _rankable(truth, scores)
    if rankable is None:
        return float("nan")
    truth, scores = rankable

    ranks = _midranks(scores)
    positives = int(np.count_nonzero(truth))
    negatives = truth.size - positives
    won = float(ranks[truth].sum()) - 0.5 * positives * (positives + 1)
    return won / (positives * negatives)


def _midranks(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank ascending from 1, giving each tied block the mean of its ranks."""
    order = np.argsort(scores, kind="stable")
    ascending = scores[order]
    opens = np.ones(ascending.size, dtype=bool)
    opens[1:] = ascending[1:] != ascending[:-1]

    starts = np.flatnonzero(opens)
    stops = np.append(starts[1:], ascending.size)
    # Positions start..stop-1 hold the 1-based ranks start+1..stop, whose mean is
    # this. Every member of the block takes it.
    shared = (starts + stops + 1) / 2.0
    block = np.cumsum(opens) - 1

    ranks = np.empty(ascending.size, dtype=np.float64)
    ranks[order] = shared[block]
    return ranks
