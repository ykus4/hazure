"""Running several detectors over the same series and putting the numbers together.

Nothing here that the metrics in :mod:`hazure.evaluation` do not already do. What
it saves is the loop, and the two mistakes that loop invites: scoring one detector
event-based and the next sample-based, and quoting recall without quoting what it
cost in alerts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hazure.evaluation import detection_delay, f1_score, precision, recall
from hazure.events import to_events

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hazure import BaseDetector
    from hazure.datasets.dataset import Dataset


__all__ = [
    "compare",
]


def compare(
    detectors: Mapping[str, BaseDetector],
    dataset: Dataset,
    *,
    thresh: float = 0.5,
    fit_on: Any = None,
) -> dict[str, dict[str, float]]:
    """Score several detectors against one dataset's ground truth.

    Parameters
    ----------
    detectors
        Detectors to compare, keyed by whatever name should appear in the result.
        Each is used as given, so anything already fitted stays fitted.
    dataset
        What to run them on, from :func:`~hazure.datasets.make_series`,
        :func:`~hazure.datasets.load_nab`, or assembled by hand.
    thresh
        Coverage threshold for the event-based metrics: the fraction of an event
        that must be covered for it to count. In ``(0, 1]``.
    fit_on
        Fit each detector on this before running it, instead of on ``dataset.data``
        itself. Pass a clean stretch of history to measure the "learn normal, then
        watch" setup rather than the unsupervised one.

    Returns
    -------
    dict of dict
        One entry per detector, holding ``"recall"``, ``"precision"``, ``"f1"``
        (all event-based), ``"alerts"`` (how many distinct alerts it raised) and
        ``"delay"`` (mean time from the start of an event to its first flagged
        sample, in seconds, or ``nan`` when nothing was caught).

    Raises
    ------
    TypeError
        ``detectors`` is not a mapping, or a value in it is not a detector.
    ValueError
        A detector returned labels for more than one column.

    See Also
    --------
    hazure.evaluation : The metrics this is a loop over.
    hazure.budget_threshold : Choosing a cut-off from what ``alerts`` costs.

    Notes
    -----
    ``alerts`` is there because the other three can all be bought with it. A
    detector that flags a third of the series will catch everything, and the
    event-based precision that ought to punish it often does not — a wide alert
    overlapping a true event counts as justified. Read the four together.

    Examples
    --------
    Three detectors against three planted level shifts. Only one of them is built
    to see a change in the mean at all:

    >>> from hazure import IqrDetector, LevelShiftDetector, SpikeDetector
    >>> from hazure.datasets import compare, make_series
    >>> dataset = make_series("level_shift", n=2000, n_anomalies=3, strength=5.0)
    >>> table = compare(
    ...     {
    ...         "iqr": IqrDetector(),
    ...         "spike": SpikeDetector(window=24),
    ...         "shift": LevelShiftDetector(window=24),
    ...     },
    ...     dataset,
    ... )
    >>> table["shift"]["recall"], table["iqr"]["recall"]
    (1.0, 0.0)
    """
    if not hasattr(detectors, "items"):
        msg = (
            f"compare() needs a mapping from name to detector, got "
            f"{type(detectors).__name__}."
        )
        raise TypeError(msg)

    truth = dataset.events
    return {
        name: _score_one(detector, dataset.data, truth, thresh, fit_on)
        for name, detector in detectors.items()
    }


def _score_one(
    detector: BaseDetector,
    data: Any,
    truth: Any,
    thresh: float,
    fit_on: Any,
) -> dict[str, float]:
    """Run one detector and reduce its labels to the four numbers plus the count.

    Parameters
    ----------
    detector
        The detector.
    data
        What to detect on.
    truth
        Ground truth intervals.
    thresh
        Coverage threshold for the event-based metrics.
    fit_on
        What to fit on, or None to fit on ``data``.

    Returns
    -------
    dict of float
        The scores for this detector.

    Raises
    ------
    TypeError
        It is not a detector.
    ValueError
        It returned labels for more than one column.
    """
    if not hasattr(detector, "fit_detect"):
        msg = (
            f"compare() needs detectors, and {type(detector).__name__} has no "
            f"fit_detect(). Pair a scorer with a threshold using ScoreDetector."
        )
        raise TypeError(msg)

    if fit_on is None:
        labels = detector.fit_detect(data)
    else:
        labels = detector.fit(fit_on).detect(data)

    found = to_events(labels)
    if isinstance(found, dict):
        msg = (
            f"compare() scores one series at a time, but {type(detector).__name__} "
            f"returned labels for {sorted(found)}. Pass a single-column series."
        )
        raise ValueError(msg)

    delay = cast("float", detection_delay(truth, found))
    return {
        "recall": cast("float", recall(truth, found, thresh)),
        "precision": cast("float", precision(truth, found, thresh)),
        "f1": cast("float", f1_score(truth, found, thresh, thresh)),
        "alerts": float(found.n_events),
        # Delays come back in nanoseconds, which is not a unit anyone reads.
        "delay": delay / 1e9,
    }
