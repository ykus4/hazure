"""How long each outage ran before anything was said about it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from hazure.evaluation.metrics import _dispatch, _joined
from hazure.events import Events, to_events

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "detection_delay",
    "detection_delays",
]


#: Reductions :func:`detection_delay` will apply to the per-event delays.
_STATISTICS = ("mean", "median", "max")


def detection_delays(
    y_true: Any, y_pred: Any
) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
    """Time from each true event's start to the first prediction inside it.

    Precision and recall say whether an outage was caught; they never say when.
    This does, one figure per true event, so that a detector which finds
    everything six hours late stops looking perfect.

    Parameters
    ----------
    y_true
        Ground truth. An ``Events``, a label series or frame that
        :func:`~hazure.events.to_events` can convert, a list of timestamps and
        ``(start, end)`` pairs, or a dict of any of those.
    y_pred
        Predictions, in the same form as ``y_true``. Both must be point-based or
        both event-based.

    Returns
    -------
    numpy.ndarray or dict of numpy.ndarray
        One delay in nanoseconds per true event, in event order, as ``float64``
        so that a never-detected event can be ``nan``. A dict input gives a dict
        of arrays keyed the same way. Convert one entry to a duration with
        ``numpy.timedelta64(int(d), "ns")``, or divide the array by the number of
        nanoseconds in the unit you want to read.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        The two label series sit on different time axes, or the two inputs carry
        different column names or keys.

    See Also
    --------
    detection_delay : The same delays, reduced to one number.
    hazure.evaluation.recall : Whether each event was caught at all.

    Notes
    -----
    An event nobody ever overlapped is ``nan``, not zero and not the event's own
    duration. Both of those would be lies that average into a summary; a missed
    event has no delay, and :func:`~hazure.evaluation.recall` is the metric that
    counts it.

    A prediction that opens before the true event scores ``0``, never a negative
    number. The delay measures the first moment the prediction and the event
    coincide, and that moment cannot precede the event: firing early is not being
    early to an outage that had not started, it is a false positive that happens
    to run into one.

    Label series are converted with :func:`~hazure.events.to_events`, so a run
    of consecutive flags on a regular axis is one event rather than several, and
    the delay is measured from the run's first sample.

    Examples
    --------
    A three-hour outage opening at 01:00, first flagged at 03:00, is two hours
    late:

    >>> import numpy as np, pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=6, freq="h")
    >>> truth = pd.Series([0, 1, 1, 1, 0, 0], index=index)
    >>> guess = pd.Series([0, 0, 0, 1, 0, 0], index=index)
    >>> delays = detection_delays(truth, guess)
    >>> delays
    array([7.2e+12])
    >>> str(np.timedelta64(int(delays[0]), "ns").astype("m8[m]"))
    '120 minutes'

    The second event here is never touched, so its delay is unknown rather than
    large:

    >>> detection_delays([(0, 100), (200, 300)], [(50, 60)])
    array([50., nan])
    """
    return _dispatch(y_true, y_pred, _event_delays, _event_delays, align=_as_events)


def detection_delay(
    y_true: Any, y_pred: Any, *, statistic: str = "mean"
) -> float | dict[str, float]:
    """One summary of :func:`detection_delays`, over the events that were caught.

    Read this next to a recall, always. The delays of undetected events are not
    reported here — they do not exist — so a detector that catches one outage out
    of fifty, quickly, posts a better delay than one that catches all fifty a
    little more slowly. The pair of numbers is the honest description; either one
    alone can be made to look good.

    Parameters
    ----------
    y_true
        Ground truth, in any form :func:`detection_delays` accepts.
    y_pred
        Predictions, in the same form as ``y_true``.
    statistic
        How to reduce the per-event delays: ``"mean"``, ``"median"`` or
        ``"max"``. The median resists a single pathological event; the max is the
        worst case, which is often what an SLA is written against.

    Returns
    -------
    float or dict of float
        The reduction, in nanoseconds, or one per column / dict key. ``nan`` when
        no true event was detected at all, and ``nan`` when there were no true
        events to detect.

    Raises
    ------
    TypeError
        The two arguments are not both point-based or both event-based.
    ValueError
        ``statistic`` is not one of the three above, the two label series sit on
        different time axes, or the two inputs carry different column names or
        keys.

    See Also
    --------
    detection_delays : The per-event delays this reduces.

    Notes
    -----
    Undetected events are dropped before the reduction rather than counted as
    zero or as infinite, for the reason given in :func:`detection_delays`.

    Examples
    --------
    Two events caught 50 ns in, one missed entirely:

    >>> truth = [(0, 100), (200, 300), (400, 500)]
    >>> guess = [(50, 60), (250, 260)]
    >>> detection_delay(truth, guess)
    50.0

    The missed event does not drag the average up, so quote the recall with it:

    >>> from hazure.evaluation import recall
    >>> recall(truth, guess, thresh=0.1)
    0.6666666666666666

    An hourly example, read back as a duration:

    >>> import numpy as np, pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=6, freq="h")
    >>> truth = pd.Series([0, 1, 1, 1, 0, 0], index=index)
    >>> guess = pd.Series([0, 0, 1, 1, 0, 0], index=index)
    >>> str(np.timedelta64(int(detection_delay(truth, guess)), "ns").astype("m8[h]"))
    '1 hours'
    """
    if statistic not in _STATISTICS:
        msg = (
            f"statistic must be one of {list(_STATISTICS)}, got {statistic!r}. "
            f"Use detection_delays for the individual delays."
        )
        raise ValueError(msg)
    return _dispatch(
        y_true,
        y_pred,
        _event_delay,
        _event_delay,
        align=_as_events,
        statistic=statistic,
    )


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _as_events(truth: TimeSeries, guess: TimeSeries) -> tuple[Events, Events]:
    """Read a pair of label series as two event sets on one time axis.

    A delay is a distance between instants, so unlike the counting metrics these
    two functions have no separate point-based kernel: label input is converted
    and scored as intervals. The shared-axis check still applies, and matters
    more here than elsewhere — period semantics stretch a flag to the sampling
    interval, so two series on different axes would be measured with different
    rulers.
    """
    _joined(truth, guess)
    # Each side is one column by the time it reaches here, so to_events returns
    # a bare Events rather than a dict of them.
    return (
        cast("Events", to_events(truth)),
        cast("Events", to_events(guess)),
    )


def _event_delays(truth: Events, guess: Events) -> NDArray[np.float64]:
    """Nanoseconds from each true event's start to its first overlap."""
    delays = np.full(truth.n_events, np.nan, dtype=np.float64)
    overlap = truth.intersect(guess)
    if overlap.n_events == 0:
        return delays

    # Every piece of the intersection lies inside exactly one true event, and
    # the pieces are sorted by start, so the first piece an event owns is its
    # earliest overlap. np.unique reports the first index of each owner.
    owner = np.searchsorted(truth.bounds[:, 0], overlap.bounds[:, 0], side="right") - 1
    detected, first = np.unique(owner, return_index=True)
    # The intersection is clipped to the true event, so its start is never
    # before the event's: a prediction that opened earlier lands on 0 here
    # rather than going negative.
    delays[detected] = overlap.bounds[first, 0] - truth.bounds[detected, 0]
    return delays


def _event_delay(truth: Events, guess: Events, *, statistic: str = "mean") -> float:
    """Reduce :func:`_event_delays` over the events that were detected."""
    delays = _event_delays(truth, guess)
    seen = delays[~np.isnan(delays)]
    if seen.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(seen.mean())
    if statistic == "median":
        return float(np.median(seen))
    return float(seen.max())
