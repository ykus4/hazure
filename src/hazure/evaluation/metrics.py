"""Precision, recall, F1 and IoU, over samples or over intervals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from hazure import TimeSeries
from hazure.events import Events

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "f1_score",
    "iou",
    "precision",
    "recall",
]


_TRUE_COLUMN = "y_true"


_PRED_COLUMN = "y_pred"


def recall(y_true: Any, y_pred: Any, thresh: float = 0.5) -> float | dict[str, float]:
    """Fraction of true anomalies that were detected.

    Also known as sensitivity, hit rate, or the true positive rate.

    Parameters
    ----------
    y_true
        Ground truth. A label series or frame, an ``Events``, a list of
        timestamps and ``(start, end)`` pairs, or a dict of any of those.
    y_pred
        Predictions, in the same form as ``y_true``. Both must be point-based or
        both event-based.
    thresh
        Event-based only: the fraction of a true event's duration that the
        prediction must cover for it to count as detected. In ``(0, 1]``. An
        event covering *k* samples of a regular series lasts exactly *k* steps,
        so a threshold of ``k / m`` means "at least *k* of the event's *m*
        samples".

    Returns
    -------
    float or dict of float
        One score, or one per column / dict key. ``nan`` when ``y_true`` holds
        no anomalies at all.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        ``thresh`` is outside ``(0, 1]``, the two label series sit on different
        time axes, or the two inputs carry different column names or keys.

    Examples
    --------
    Event-based: a four-hour outage with three of its hours flagged is covered
    well past the default threshold, so the outage counts as detected.

    >>> import pandas as pd
    >>> from hazure.events import to_events
    >>> index = pd.date_range("2024-01-01", periods=5, freq="h")
    >>> truth = pd.Series([1, 1, 1, 1, 0], index=index)
    >>> guess = pd.Series([1, 1, 1, 0, 0], index=index)
    >>> recall(to_events(truth), to_events(guess))
    1.0

    Coverage is exactly three of the four hours, so demanding more misses it:

    >>> recall(to_events(truth), to_events(guess), 0.8)
    0.0

    Point-based, on the same labels: three of the four true points were found.

    >>> recall(truth, guess)
    0.75
    """
    _check_thresh(thresh, "thresh")
    return _dispatch(y_true, y_pred, _point_recall, _event_recall, thresh=thresh)


def precision(
    y_true: Any, y_pred: Any, thresh: float = 0.5
) -> float | dict[str, float]:
    """Fraction of detections that were real.

    This is :func:`recall` with the arguments swapped: a predicted event counts
    as correct when enough of *its* duration is covered by the ground truth.

    Parameters
    ----------
    y_true
        Ground truth, in any form :func:`recall` accepts.
    y_pred
        Predictions, in the same form as ``y_true``.
    thresh
        Event-based only: the fraction of a predicted event's duration that the
        ground truth must cover for it to count as correct. In ``(0, 1]``.

    Returns
    -------
    float or dict of float
        One score, or one per column / dict key. ``nan`` when ``y_pred`` holds
        no anomalies at all.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        ``thresh`` is outside ``(0, 1]``, the two label series sit on different
        time axes, or the two inputs carry different column names or keys.

    Examples
    --------
    The prediction lies entirely inside the true outage, so it is fully
    justified:

    >>> import pandas as pd
    >>> from hazure.events import to_events
    >>> index = pd.date_range("2024-01-01", periods=5, freq="h")
    >>> truth = pd.Series([1, 1, 1, 1, 0], index=index)
    >>> guess = pd.Series([1, 1, 1, 0, 0], index=index)
    >>> precision(to_events(truth), to_events(guess))
    1.0

    Predicting a second, spurious event halves it:

    >>> noisy = pd.Series([1, 1, 0, 0, 1], index=index)
    >>> precision(to_events(truth), to_events(noisy))
    0.5

    Point-based, on the same labels: every flagged point was a true one.

    >>> precision(truth, guess)
    1.0
    """
    _check_thresh(thresh, "thresh")
    return recall(y_pred, y_true, thresh)


def f1_score(
    y_true: Any,
    y_pred: Any,
    recall_thresh: float = 0.5,
    precision_thresh: float = 0.5,
) -> float | dict[str, float]:
    """Harmonic mean of :func:`precision` and :func:`recall`.

    The two thresholds are independent because they answer different questions:
    how much of a true event must be covered to count as caught, and how much of
    an alert must be real to count as justified. A monitoring team that tolerates
    broad alerts but wants outages caught early will set them differently.

    Parameters
    ----------
    y_true
        Ground truth, in any form :func:`recall` accepts.
    y_pred
        Predictions, in the same form as ``y_true``.
    recall_thresh
        Coverage threshold passed to :func:`recall`. In ``(0, 1]``.
    precision_thresh
        Coverage threshold passed to :func:`precision`. In ``(0, 1]``.

    Returns
    -------
    float or dict of float
        One score, or one per column / dict key. ``nan`` when precision and
        recall are both zero, or when either is itself undefined.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        A threshold is outside ``(0, 1]``, the two label series sit on different
        time axes, or the two inputs carry different column names or keys.

    Examples
    --------
    The one true outage is caught and the one alert is justified:

    >>> import pandas as pd
    >>> from hazure.events import to_events
    >>> index = pd.date_range("2024-01-01", periods=5, freq="h")
    >>> truth = pd.Series([1, 1, 1, 1, 0], index=index)
    >>> guess = pd.Series([1, 1, 1, 0, 0], index=index)
    >>> f1_score(to_events(truth), to_events(guess))
    1.0

    Point-based, on the same labels: recall 0.75 against precision 1.0.

    >>> round(f1_score(truth, guess), 4)
    0.8571
    """
    _check_thresh(recall_thresh, "recall_thresh")
    _check_thresh(precision_thresh, "precision_thresh")
    recalled = recall(y_true, y_pred, recall_thresh)
    precise = precision(y_true, y_pred, precision_thresh)
    # Both calls see the same input shape, so either both are dicts over the
    # same keys or neither is.
    if isinstance(recalled, dict):
        paired = cast("dict[str, float]", precise)
        return {key: _harmonic(recalled[key], paired[key]) for key in recalled}
    return _harmonic(recalled, cast("float", precise))


def iou(y_true: Any, y_pred: Any) -> float | dict[str, float]:
    """Intersection over union of the anomalous regions.

    Unlike the three metrics above this has no threshold: it measures agreement
    directly, as the size of the region both inputs call anomalous over the size
    of the region either of them does. Point-based inputs count samples;
    event-based inputs measure duration.

    Parameters
    ----------
    y_true
        Ground truth, in any form :func:`recall` accepts.
    y_pred
        Predictions, in the same form as ``y_true``.

    Returns
    -------
    float or dict of float
        One score in ``[0, 1]``, or one per column / dict key. ``nan`` when
        neither input marks anything anomalous, so the union is empty.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        The two label series sit on different time axes, or the two inputs carry
        different column names or keys.

    Examples
    --------
    Event-based: three of the four anomalous hours agree, and the union is all
    four.

    >>> import pandas as pd
    >>> from hazure.events import to_events
    >>> index = pd.date_range("2024-01-01", periods=5, freq="h")
    >>> truth = pd.Series([1, 1, 1, 1, 0], index=index)
    >>> guess = pd.Series([1, 1, 1, 0, 0], index=index)
    >>> iou(to_events(truth), to_events(guess))
    0.75

    Point-based on the same labels gives the same number, because a duration
    ratio over period events and a count of samples are the same measurement.

    >>> iou(truth, guess)
    0.75
    """
    return _dispatch(y_true, y_pred, _point_iou, _event_iou)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_EVENT_LIKE = (Events, list, tuple)


def _dispatch(
    y_true: Any,
    y_pred: Any,
    on_points: Any,
    on_events: Any,
    **options: Any,
) -> float | dict[str, float]:
    """Route one pair of inputs to the point-based or event-based kernel.

    Recursion happens here rather than inside each metric, which is what keeps
    ``options`` intact all the way down to the leaves.
    """
    true_kind = _kind(y_true)
    pred_kind = _kind(y_pred)
    if true_kind != pred_kind:
        msg = (
            f"y_true is {true_kind} but y_pred is {pred_kind}. Both must be "
            f"label series, both Events (or lists of intervals), or both dicts."
        )
        raise TypeError(msg)

    if true_kind == "mapping":
        _check_keys(set(y_true), set(y_pred), "key")
        scores: dict[str, float] = {}
        for key in y_true:
            score = _dispatch(y_true[key], y_pred[key], on_points, on_events, **options)
            if isinstance(score, dict):
                msg = (
                    f"Key {key!r} must hold a single anomaly type, but it "
                    f"expanded into {sorted(score)}. Pass a one-column label "
                    f"series, an Events, or a list of intervals."
                )
                raise TypeError(msg)
            scores[key] = score
        return scores

    if true_kind == "events":
        return float(
            on_events(Events.from_any(y_true), Events.from_any(y_pred), **options)
        )

    truth = TimeSeries.from_any(y_true)
    guess = TimeSeries.from_any(y_pred)
    if truth.n_columns > 1 or guess.n_columns > 1:
        _check_keys(set(truth.columns), set(guess.columns), "column")
        return {
            name: float(
                on_points(*_aligned(truth.select(name), guess.select(name)), **options)
            )
            for name in truth.columns
        }
    return float(on_points(*_aligned(truth, guess), **options))


def _kind(value: Any) -> str:
    """Classify an input as ``mapping``, ``events`` or ``labels``."""
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, _EVENT_LIKE):
        return "events"
    return "labels"


def _check_keys(left: set[str], right: set[str], noun: str) -> None:
    """Require both inputs to describe the same set of anomaly types."""
    if left != right:
        msg = (
            f"y_true and y_pred must describe the same {noun}s, but "
            f"{sorted(left ^ right)} appear in only one of them."
        )
        raise ValueError(msg)


def _aligned(
    truth: TimeSeries, guess: TimeSeries
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the two label columns as booleans on one shared time axis.

    The join is the check: an outer join on time only grows when the two axes
    disagree, so a longer result means the caller was about to compare samples
    that never lined up.
    """
    joined = truth.wrap(truth.values, [_TRUE_COLUMN]).join(
        guess.wrap(guess.values, [_PRED_COLUMN])
    )
    if joined.n_rows != truth.n_rows or joined.n_rows != guess.n_rows:
        msg = (
            f"The two label series must share a time axis, but the union of "
            f"their axes has {joined.n_rows} timestamps where the series have "
            f"{truth.n_rows} and {guess.n_rows}. Reindex or resample them onto "
            f"one axis first."
        )
        raise ValueError(msg)
    return (
        _binary(joined.column_values(_TRUE_COLUMN)),
        _binary(joined.column_values(_PRED_COLUMN)),
    )


def _binary(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Read a label column as booleans, treating NaN as not anomalous."""
    return np.asarray(np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) == 1.0)


def _check_thresh(thresh: Any, name: str) -> None:
    """Require a coverage threshold in ``(0, 1]``."""
    if not isinstance(thresh, float | int) or isinstance(thresh, bool):
        msg = f"{name} must be a number, got {type(thresh).__name__}."
        raise TypeError(msg)
    if not 0.0 < thresh <= 1.0:
        msg = (
            f"{name} must be greater than 0 and at most 1, got {thresh}. It is "
            f"the fraction of an event's duration that must be covered."
        )
        raise ValueError(msg)


def _harmonic(left: float, right: float) -> float:
    """Harmonic mean, or nan when it is undefined."""
    total = left + right
    if not np.isfinite(total) or total == 0.0:
        return float("nan")
    return 2.0 * left * right / total


# ---------------------------------------------------------------------------
# point-based kernels
# ---------------------------------------------------------------------------


def _point_recall(
    truth: NDArray[np.bool_], guess: NDArray[np.bool_], *, thresh: float = 0.5
) -> float:
    """Divide detected true points by true points.

    ``thresh`` is accepted so that the point-based and event-based kernels
    share one signature, and ignored because a point has no duration to
    cover partially.
    """
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return float("nan")
    return int(np.count_nonzero(truth & guess)) / positives


def _point_iou(truth: NDArray[np.bool_], guess: NDArray[np.bool_]) -> float:
    """Points flagged by both over points flagged by either."""
    union = int(np.count_nonzero(truth | guess))
    if union == 0:
        return float("nan")
    return int(np.count_nonzero(truth & guess)) / union


# ---------------------------------------------------------------------------
# event-based kernels
# ---------------------------------------------------------------------------


def _event_recall(truth: Events, guess: Events, *, thresh: float = 0.5) -> float:
    """Fraction of true events covered by at least ``thresh`` of their duration.

    One rule covers every event, instantaneous ones included: an instant lasts
    1 ns, so a prediction containing it covers all of it and one that misses it
    covers none.
    """
    total = truth.n_events
    if total == 0:
        return float("nan")

    covered: NDArray[np.float64] = np.zeros(total, dtype=np.float64)
    overlap = truth.intersect(guess)
    if overlap.n_events:
        # Every piece of the intersection lies inside exactly one true event,
        # because both sets are disjoint, so the owning event is the last one
        # whose start is at or before the piece's.
        owner = (
            np.searchsorted(truth.bounds[:, 0], overlap.bounds[:, 0], side="right") - 1
        )
        covered = np.asarray(
            np.bincount(
                owner,
                weights=overlap.durations.astype(np.float64),
                minlength=total,
            ),
            dtype=np.float64,
        )

    # thresh is strictly positive and every duration is at least 1 ns, so an
    # uncovered event can never clear the bar.
    hits = covered >= thresh * truth.durations
    return int(np.count_nonzero(hits)) / total


def _event_iou(truth: Events, guess: Events) -> float:
    """Intersected duration over union duration.

    Every event lasts at least 1 ns, so a zero union means there were no events
    at all — the one case where the ratio is genuinely undefined.
    """
    union = truth.union(guess).total_duration
    if union == 0:
        return float("nan")
    return truth.intersect(guess).total_duration / union
