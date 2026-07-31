"""Choosing where to draw the line.

A scorer says how unusual each point is. Turning that into an alert needs a
number, and picking that number is the question people actually get stuck on:
every threshold in :mod:`hazure.thresholds` is parameterised by *something*, and
none of those somethings is "the answer I want".

Two ways out of it, depending on what you have.

If you have labelled history — even a handful of incidents somebody wrote down —
:func:`tune_threshold` searches for the cut-off that scores best against them.
Event-based F1 by default, because a monitor is judged on outages caught and
pages sent, not on samples classified.

If you have no labels, which is the case this library is mostly about, you still
have an alert budget: whatever the metric does, nobody is going to look at more
than one page a week. :func:`budget_threshold` finds the most sensitive cut-off
that stays inside that budget, which turns an unanswerable question about
distributions into an answerable one about how much attention you have.

Both hand back a :class:`Calibration`, which carries the whole curve it searched
and not only the winner — a threshold chosen off a flat peak is worth knowing
about, and so is one chosen off a spike.

Hyperparameters other than the cut-off need no machinery from here: components
follow the scikit-learn parameter conventions closely enough for
``sklearn.model_selection.GridSearchCV`` to drive them directly, and
:func:`~hazure.split_train_test` supplies folds that respect time order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from hazure._core import TimeSeries, parse_duration
from hazure.evaluation import f1_score, iou, precision, recall
from hazure.events import Events, expand_events, to_events
from hazure.thresholds import FixedThreshold

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import timedelta

    from numpy.typing import NDArray


__all__ = [
    "Calibration",
    "budget_threshold",
    "tune_threshold",
]


#: Metrics :func:`tune_threshold` will look up by name. Each is maximised, which
#: is why ``iou`` belongs here and a delay does not.
_METRICS: dict[str, Callable[..., Any]] = {
    "f1": f1_score,
    "iou": iou,
    "precision": precision,
    "recall": recall,
}


@dataclass(frozen=True, slots=True)
class Calibration:
    """Where the line was drawn, what that achieved, and what else was considered.

    Attributes
    ----------
    cut_off : float
        The chosen cut-off. Scores strictly above it are anomalous.
    score : float
        What :attr:`objective` came to at :attr:`cut_off`.
    objective : str
        What was being optimised, as text — ``"event f1"``, ``"alerts per 7d"``.
        Worth carrying, because a bare number two functions could both have
        produced is not self-describing.
    curve : tuple of tuple
        Every ``(cut_off, score)`` pair examined, in increasing order of cut-off.
        This is the part worth looking at: a peak the width of one candidate is a
        peak fitted to noise, and no summary statistic will tell you that.

    See Also
    --------
    tune_threshold : Choose a cut-off against labelled history.
    budget_threshold : Choose a cut-off against an alert budget.

    Examples
    --------
    >>> calibration = Calibration(2.5, 0.8, "event f1", ((1.0, 0.5), (2.5, 0.8)))
    >>> calibration.threshold
    FixedThreshold(high=2.5)
    """

    cut_off: float
    score: float
    objective: str
    curve: tuple[tuple[float, float], ...]

    @property
    def threshold(self) -> FixedThreshold:
        """The cut-off as a threshold, ready to pair with a scorer.

        Returns
        -------
        FixedThreshold
            A one-sided fence at :attr:`cut_off`. Pair it with the scorer the
            calibration was computed from, through
            :class:`~hazure.ScoreDetector`.
        """
        return FixedThreshold(high=self.cut_off)

    def __repr__(self) -> str:
        return (
            f"Calibration(cut_off={self.cut_off:.6g}, {self.objective}="
            f"{self.score:.6g}, over {len(self.curve)} candidates)"
        )


def tune_threshold(
    y_true: Any,
    scores: Any,
    *,
    metric: str | Callable[..., Any] = "f1",
    candidates: int = 200,
    events: bool = True,
    **options: Any,
) -> Calibration | dict[str, Calibration]:
    """Find the cut-off that scores best against labelled history.

    Parameters
    ----------
    y_true
        Ground truth: a label series, an :class:`~hazure.Events`, a list of
        intervals, or a dict of those keyed by score column. A single ground
        truth given against several score columns is used for all of them.
    scores
        Continuous scores from any :class:`~hazure.BaseScorer`, on the same time
        axis as ``y_true``. Higher means more anomalous.
    metric
        What to maximise: ``"f1"``, ``"iou"``, ``"precision"``, ``"recall"``, or
        any callable taking ``(y_true, y_pred)`` and returning a float. Callables
        receive whatever ``events`` selected, so a custom one has to accept the
        same kind of object.
    candidates
        Roughly how many cut-offs to try. They are placed at quantiles of the
        scores rather than at evenly spaced values, half evenly and half packed
        towards the largest score, so the search resolves the extreme tail where
        a usable fence normally sits.
    events
        Score by interval rather than by sample. This is the default because it
        is the question a monitor is judged on — an outage caught late is caught,
        and a hundred alerts inside one incident are one page. Turn it off to
        weight every sample equally.
    **options
        Passed through to the metric, e.g. ``thresh=0.8`` to demand that 80% of
        an event's duration be covered before it counts as caught.

    Returns
    -------
    Calibration or dict of Calibration
        The chosen cut-off and the curve behind it, or one per score column.

    Raises
    ------
    KeyError
        ``metric`` names something not in the registry.
    TypeError
        ``metric`` is neither a name nor a callable.
    ValueError
        ``candidates`` is below 2, the scores hold nothing but ``NaN``, or
        ``y_true`` cannot be matched up with the score columns.

    See Also
    --------
    budget_threshold : The same question without labels, answered from an alert
        budget instead.

    Notes
    -----
    Ties go to the higher cut-off. A flat top to the curve is common — the metric
    cannot tell apart two fences with no scores between them — and of two equally
    good fences the higher one raises fewer alerts, so it is the one that survives
    contact with people.

    A metric that comes back undefined, as precision does when nothing at all is
    flagged, is recorded in the curve as ``nan`` and never chosen.

    None of the four metrics counts alerts, and event-based precision in particular
    does not punish a fragmented one: two alerts inside the same true event are two
    justified alerts. So a cut-off with a perfect F1 can still page twice for one
    incident. Check :func:`~hazure.to_events` on the result, or debounce with
    :func:`~hazure.expand_events`, or calibrate with :func:`budget_threshold`
    instead, which counts alerts by construction.

    The cut-off this finds is fitted to ``y_true``, and quoting the same metric at
    the same cut-off on the same data would be reporting a training score. Tune on
    one fold and measure on the next; :func:`~hazure.split_train_test` exists for
    that.

    Examples
    --------
    A series with two planted outages, scored by deviation from the median, and a
    cut-off chosen against the intervals somebody wrote down:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hazure import DeviationScorer
    >>> index = pd.date_range("2024-01-01", periods=480, freq="h", name="time")
    >>> rng = np.random.default_rng(0)
    >>> values = pd.Series(rng.normal(0, 1, 480), index=index, name="x")
    >>> values.iloc[100:106] += 6.0
    >>> values.iloc[300:304] += 5.0
    >>> truth = pd.Series(0.0, index=index)
    >>> truth.iloc[100:106] = 1.0
    >>> truth.iloc[300:304] = 1.0
    >>> scores = DeviationScorer().fit_score(values)
    >>> best = tune_threshold(truth, scores)
    >>> best
    Calibration(cut_off=3.96769, event f1=1, over 200 candidates)

    Both outages are caught and nothing else is flagged — but the fence cuts one of
    them into two alerts, and an event-based F1 of 1.0 does not notice, because both
    fragments lie inside a true event and so both count as justified:

    >>> labels = best.threshold.apply(scores)
    >>> to_events(labels).n_events
    3

    That is worth knowing before trusting the number. Read the alert count next to
    it, or debounce with :func:`~hazure.expand_events` first.
    """
    _check_candidates(candidates)
    measure = _resolve_metric(metric)
    label = f"{'event' if events else 'sample'} {_name_of(metric)}"

    results = {
        name: _tune_column(truth, column, measure, candidates, events, label, options)
        for name, truth, column in _pairs(y_true, scores)
    }
    return _unwrap(results)


def budget_threshold(
    scores: Any,
    *,
    alerts: float = 1.0,
    per: str | timedelta | np.timedelta64 = "7d",
    gap: int | str | timedelta | np.timedelta64 | None = None,
    candidates: int = 200,
) -> Calibration | dict[str, Calibration]:
    """Find the most sensitive cut-off that stays inside an alert budget.

    No labels needed, which is the point. What is known instead is how much
    attention exists: one page a week, five a day. This lowers the fence as far as
    that allows and no further.

    Parameters
    ----------
    scores
        Continuous scores from any :class:`~hazure.BaseScorer`, over a stretch of
        history long enough to contain several budget periods. Higher means more
        anomalous.
    alerts
        How many alerts are affordable per ``per``. May be fractional — ``0.5``
        with ``per="7d"`` is one a fortnight.
    per
        The period the budget is expressed over: ``"7d"``, ``"1d"``, ``"1h"``, a
        :class:`~datetime.timedelta`, or a :class:`numpy.timedelta64`.
    gap
        Alerts closer together than this are one alert. Without it a flagged
        stretch broken by a single quiet sample counts twice, which overstates the
        rate a monitor would actually produce, since real ones debounce. A
        duration, or an ``int`` of nanoseconds.
    candidates
        Roughly how many cut-offs to try, placed at quantiles of the scores and
        packed towards the largest of them — an alert budget of one a week lives in
        the top fraction of a percent, which an evenly spaced grid cannot resolve.

    Returns
    -------
    Calibration or dict of Calibration
        The chosen cut-off, the alert rate it realises, and the whole rate curve,
        or one per score column. :attr:`Calibration.score` is alerts per ``per``.

    Raises
    ------
    ValueError
        ``alerts`` is not positive, ``per`` is not a positive duration,
        ``candidates`` is below 2, the scores hold nothing but ``NaN``, or the
        series is too short to measure a rate over.

    See Also
    --------
    tune_threshold : The same question when there is labelled history to aim at.
    hazure.expand_events : What ``gap`` is doing, available on its own.

    Notes
    -----
    The search descends. It starts at the highest candidate — which flags nothing
    and so costs nothing — and lowers the fence one candidate at a time, stopping
    at the first cut-off that breaks the budget and taking the last one that did
    not.

    Descending rather than scanning for the lowest cut-off that happens to fit is
    not a detail. Alert *count* is not monotone in the cut-off: raising a fence
    removes flagged samples, and removing one from the middle of a flagged stretch
    splits one alert into two. Taken far enough that reverses completely — a fence
    at the 1st percentile flags almost every sample, which merges into a single
    enormous alert and satisfies any budget you like. Descending until the budget
    breaks never reaches that region, because the count passes through the budget
    on the way down long before the alerts begin to merge.

    The rate is measured over the span the scores cover, so it is an estimate from
    one sample of history and inherits everything that was unusual about that
    history. A budget met over a quiet fortnight will be missed in a busy one.

    Examples
    --------
    Three weeks of a metric with a handful of excursions in it, and a budget of one
    alert a week:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hazure import DeviationScorer
    >>> index = pd.date_range("2024-01-01", periods=24 * 21, freq="h", name="time")
    >>> rng = np.random.default_rng(0)
    >>> values = pd.Series(rng.normal(0, 1, len(index)), index=index, name="rps")
    >>> for start in (40, 150, 300, 380, 460):
    ...     values.iloc[start : start + 3] += 5.0
    >>> scores = DeviationScorer().fit_score(values)
    >>> chosen = budget_threshold(scores, alerts=1, per="7d")
    >>> chosen
    Calibration(cut_off=2.88988, alerts per 7d=1, over 200 candidates)

    Three alerts over three weeks is the budget, met exactly:

    >>> from hazure.events import to_events
    >>> to_events(chosen.threshold.apply(scores)).n_events
    3

    Ask for more attention and the fence comes down:

    >>> generous = budget_threshold(scores, alerts=5, per="7d")
    >>> generous.cut_off < chosen.cut_off
    True
    """
    _check_alerts(alerts)
    _check_candidates(candidates)

    period = parse_duration(per)
    margin = _as_nanoseconds(gap)
    label = f"alerts per {per if isinstance(per, str) else _render_duration(period)}"

    results = {
        name: _budget_column(column, alerts, period, margin, candidates, label)
        for name, _, column in _pairs(None, scores)
    }
    return _unwrap(results)


# ---------------------------------------------------------------------------
# per-column search
# ---------------------------------------------------------------------------


def _tune_column(
    y_true: Any,
    scores: TimeSeries,
    measure: Callable[..., Any],
    candidates: int,
    events: bool,
    label: str,
    options: dict[str, Any],
) -> Calibration:
    """Search one score column for the cut-off that maximises a metric.

    Parameters
    ----------
    y_true
        Ground truth for this column.
    scores
        The column, as a univariate series.
    measure
        The metric to maximise.
    candidates
        How many cut-offs to try.
    events
        Whether to score by interval rather than by sample.
    label
        Human-readable name of the objective.
    options
        Extra keyword arguments for the metric.

    Returns
    -------
    Calibration
        The best cut-off found, and the curve.
    """
    truth = to_events(y_true) if events and not _is_events(y_true) else y_true
    if events and isinstance(truth, dict):
        msg = (
            f"y_true expanded into several anomaly types {sorted(truth)} for one "
            f"score column. Pass a single label series, an Events, or a dict "
            f"keyed by score column name."
        )
        raise ValueError(msg)

    grid = _grid(scores.values[:, 0], candidates)
    curve: list[tuple[float, float]] = []
    best_cut, best_score = math.nan, -math.inf
    for cut in grid:
        guess = scores.wrap(_flag(scores.values[:, 0], cut))
        value = float(measure(truth, to_events(guess) if events else guess, **options))
        curve.append((cut, value))
        # Ties go to the higher cut-off, and the grid ascends, so >= suffices.
        if math.isfinite(value) and value >= best_score:
            best_cut, best_score = cut, value

    if not math.isfinite(best_score):
        best_score = math.nan
    return Calibration(best_cut, best_score, label, tuple(curve))


def _budget_column(
    scores: TimeSeries,
    alerts: float,
    period: int,
    margin: int,
    candidates: int,
    label: str,
) -> Calibration:
    """Search one score column for the lowest cut-off inside an alert budget.

    Parameters
    ----------
    scores
        The column, as a univariate series.
    alerts
        Alerts affordable per ``period``.
    period
        The budget period, in nanoseconds.
    margin
        Alerts closer than this are merged, in nanoseconds.
    candidates
        How many cut-offs to try.
    label
        Human-readable name of the objective.

    Returns
    -------
    Calibration
        The chosen cut-off, the rate it realises, and the curve.

    Raises
    ------
    ValueError
        The series covers no time, so no rate can be measured over it.
    """
    span = _span(scores)
    periods = span / period
    allowed = alerts * periods

    grid = _grid(scores.values[:, 0], candidates)
    curve = [
        (cut, _alert_count(scores, cut, margin) / periods)
        for cut in grid  # ascending, as Calibration.curve promises
    ]

    # Descend from the fence that flags nothing, and stop where the budget breaks.
    chosen, realised = curve[-1]
    for cut, rate in reversed(curve):
        if rate * periods > allowed:
            break
        chosen, realised = cut, rate
    return Calibration(chosen, realised, label, tuple(curve))


def _alert_count(scores: TimeSeries, cut: float, margin: int) -> int:
    """Count the alerts a cut-off would have raised over this series.

    Parameters
    ----------
    scores
        A univariate score series.
    cut
        The cut-off; scores strictly above it are flagged.
    margin
        Alerts closer than this are one alert, in nanoseconds.

    Returns
    -------
    int
        Number of distinct alerts.
    """
    events = to_events(scores.wrap(_flag(scores.values[:, 0], cut)))
    if not isinstance(events, Events):  # pragma: no cover - the input is univariate
        msg = "Scoring a budget needs one score column at a time."
        raise TypeError(msg)
    if margin > 0:
        # Widening each alert by the margin and re-merging is exactly debouncing:
        # two alerts within the margin now overlap, and overlapping events merge.
        widened = expand_events(events, after=margin)
        events = widened if isinstance(widened, Events) else events
    return events.n_events


def _span(scores: TimeSeries) -> float:
    """Return the duration the series covers, in nanoseconds.

    Parameters
    ----------
    scores
        The series.

    Returns
    -------
    float
        Its span. A regular series covers one further sampling interval than its
        first and last timestamps are apart, because the last sample stands for
        the period it opens.

    Raises
    ------
    ValueError
        The series covers no time.
    """
    if scores.n_rows < 2:
        msg = (
            f"A rate cannot be measured over {scores.n_rows} observation(s). Pass "
            f"scores covering several budget periods."
        )
        raise ValueError(msg)
    span = int(scores.time[-1] - scores.time[0]) + (scores.freq or 0)
    if span <= 0:  # pragma: no cover - a sorted axis of 2+ rows always spans
        msg = "The scores cover no time, so no alert rate can be measured."
        raise ValueError(msg)
    return float(span)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _check_candidates(candidates: Any) -> None:
    """Reject a search grid too coarse to be a search.

    Parameters
    ----------
    candidates
        The value passed to a calibrator.

    Raises
    ------
    TypeError
        It is not an int.
    ValueError
        It is below 2.
    """
    if not isinstance(candidates, int) or isinstance(candidates, bool):
        msg = f"candidates must be an int, got {type(candidates).__name__}."
        raise TypeError(msg)
    if candidates < 2:
        msg = (
            f"candidates must be at least 2, got {candidates}; with one candidate "
            f"there is nothing to choose between."
        )
        raise ValueError(msg)


def _check_alerts(alerts: Any) -> None:
    """Reject an alert budget no fence could be calibrated against.

    Parameters
    ----------
    alerts
        The value passed to :func:`budget_threshold`.

    Raises
    ------
    TypeError
        It is not a number.
    ValueError
        It is not positive.
    """
    if not isinstance(alerts, float | int) or isinstance(alerts, bool):
        msg = f"alerts must be a number, got {type(alerts).__name__}."
        raise TypeError(msg)
    if alerts <= 0.0:
        msg = (
            f"alerts must be positive, got {alerts}. A budget of zero alerts is "
            f"met by any fence high enough to flag nothing, which is not a "
            f"calibration of anything."
        )
        raise ValueError(msg)


def _as_nanoseconds(gap: Any) -> int:
    """Read the debounce margin as nanoseconds, accepting a bare int as already so.

    Parameters
    ----------
    gap
        A duration, an int of nanoseconds, or None for no debouncing.

    Returns
    -------
    int
        The margin in nanoseconds; 0 for None.
    """
    if gap is None:
        return 0
    if isinstance(gap, int) and not isinstance(gap, bool):
        return gap
    return parse_duration(gap)


def _grid(values: NDArray[np.float64], candidates: int) -> list[float]:
    """Place cut-offs at quantiles of the scores, packed towards the top.

    Parameters
    ----------
    values
        One score column, ``NaN`` included.
    candidates
        Roughly how many cut-offs to produce. The result may be shorter, since
        cut-offs that coincide are one cut-off.

    Returns
    -------
    list of float
        Distinct cut-offs in increasing order. The largest score is always among
        them, so the grid always contains a fence that flags nothing.

    Raises
    ------
    ValueError
        Every score is ``NaN``, so there is nothing to place a fence among.

    Notes
    -----
    Half the candidates go on evenly spaced quantiles, which is what a metric
    being maximised wants — the peak can be anywhere. The other half approach the
    maximum geometrically, which is what an alert budget needs, and it needs it
    badly: a hundred evenly spaced quantiles of ten thousand scores leave the top
    percentile as a single interval, and every fence a monitor would actually
    accept is inside it. The geometric half resolves down to the individual
    largest scores, so a budget of one alert a week is reachable rather than
    rounded up to "flag nothing".

    When there are fewer distinct scores than candidates asked for, every one of
    them is a candidate and the search is exhaustive.
    """
    observed = values[~np.isnan(values)]
    if observed.size == 0:
        msg = (
            "Every score is NaN, so there is no cut-off to choose. Check that the "
            "scorer was given enough history to fill its windows."
        )
        raise ValueError(msg)

    distinct = np.unique(observed)
    if distinct.size <= candidates:
        return [float(value) for value in distinct]

    half = candidates // 2
    even = np.linspace(0.0, 1.0, candidates - half)
    # Deep enough to isolate the single largest score, so no achievable fence is
    # out of reach; 0.5 at the shallow end, where the even half takes over.
    tail = 1.0 - np.logspace(np.log10(0.5), np.log10(1.0 / observed.size), half)
    quantiles = np.unique(np.concatenate([even, tail]))
    return [float(value) for value in np.unique(np.quantile(observed, quantiles))]


def _flag(values: NDArray[np.float64], cut: float) -> NDArray[np.float64]:
    """Label scores strictly above ``cut``, leaving missing scores unknown."""
    labels = (values > cut).astype(np.float64)
    labels[np.isnan(values)] = np.nan
    return labels


def _pairs(y_true: Any, scores: Any) -> Iterator[tuple[str | None, Any, TimeSeries]]:
    """Yield one ``(name, ground truth, score column)`` triple per column to search.

    Parameters
    ----------
    y_true
        Ground truth in any accepted form, or None when there is none.
    scores
        The score series or frame.

    Yields
    ------
    tuple
        The column name (None when there is only one column), the ground truth
        narrowed to it, and the column itself.
    """
    ts = TimeSeries.from_any(scores)
    if ts.n_columns == 1:
        yield None, y_true, ts
        return
    for name in ts.columns:
        yield name, _narrow(y_true, name, ts.columns), ts.select(name)


def _narrow(y_true: Any, name: str, columns: tuple[str, ...]) -> Any:
    """Pick out the ground truth belonging to one score column.

    Parameters
    ----------
    y_true
        Ground truth in any accepted form, or None.
    name
        The score column being calibrated.
    columns
        Every score column, for the error message.

    Returns
    -------
    object
        The ground truth for ``name``.

    Raises
    ------
    ValueError
        There is no ground truth for this column.
    """
    if y_true is None or _is_events(y_true):
        # One set of intervals stands for the whole frame: several scores of the
        # same underlying incidents is the ordinary case.
        return y_true
    if isinstance(y_true, dict):
        if name not in y_true:
            msg = (
                f"y_true has no entry for score column {name!r}; it holds "
                f"{sorted(y_true)}. Every column of {list(columns)} needs one."
            )
            raise ValueError(msg)
        return y_true[name]

    truth = TimeSeries.from_any(y_true)
    if truth.n_columns == 1:
        return truth
    if name in truth.columns:
        return truth.select(name)
    msg = (
        f"y_true carries {list(truth.columns)} and has no column {name!r} to "
        f"match the scores {list(columns)} against. Pass one label series, a "
        f"frame with the same column names, or a dict keyed by them."
    )
    raise ValueError(msg)


def _is_events(value: Any) -> bool:
    """Report whether a ground truth is interval-based rather than per-sample."""
    return isinstance(value, Events | list | tuple)


def _resolve_metric(metric: Any) -> Callable[..., Any]:
    """Look up a metric by name, or accept a callable.

    Parameters
    ----------
    metric
        A name from the registry, or a callable.

    Returns
    -------
    callable
        The metric.

    Raises
    ------
    KeyError
        The name is not in the registry.
    TypeError
        It is neither a name nor callable.
    """
    if callable(metric):
        return cast("Callable[..., Any]", metric)
    if isinstance(metric, str):
        if metric not in _METRICS:
            msg = (
                f"{metric!r} is not a metric tune_threshold knows; it accepts "
                f"{sorted(_METRICS)}, or any callable taking (y_true, y_pred)."
            )
            raise KeyError(msg)
        return _METRICS[metric]
    msg = f"metric must be a name or a callable, got {type(metric).__name__}."
    raise TypeError(msg)


def _name_of(metric: str | Callable[..., Any]) -> str:
    """Name a metric for :attr:`Calibration.objective`."""
    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", "metric")


def _render_duration(nanoseconds: int) -> str:
    """Render a duration for :attr:`Calibration.objective`."""
    for unit, size in (("d", 86_400), ("h", 3_600), ("min", 60), ("s", 1)):
        scale = size * 1_000_000_000
        if nanoseconds >= scale and nanoseconds % scale == 0:
            return f"{nanoseconds // scale}{unit}"
    return f"{nanoseconds}ns"


def _unwrap(
    results: dict[str | None, Calibration],
) -> Calibration | dict[str, Calibration]:
    """Return one calibration for a single column, or a dict keyed by column."""
    if len(results) == 1 and None in results:
        return results[None]
    return {str(name): value for name, value in results.items()}
