"""Combining several label series into one.

An aggregator is how an ensemble of detectors, or a multi-condition rule,
produces a single answer. Every aggregator here emits one column named
``anomaly``.

A label is ``1.0`` anomalous, ``0.0`` normal or ``NaN`` unknown, all in a float
column. ``NaN`` is a real third state, not a missing 0: a rolling detector cannot
label the first few points of a series, and saying "unknown" there is different
from saying "normal". Combining labels is therefore three-valued logic — see
:mod:`hazure.ensemble` for the truth table each class implements.

Any non-zero label counts as anomalous, so a detector that emits counts or
confidences rather than a strict 0/1 still aggregates sensibly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hazure import BaseAggregator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "AndAggregator",
    "CustomizedAggregator",
    "OrAggregator",
    "VoteAggregator",
]


class OrAggregator(BaseAggregator):
    """Flag a point that any input flags.

    The union of the inputs: use it to collect several kinds of anomaly — a
    spike detector, a level-shift detector, a range check — into one label
    series. Unknown inputs only matter when nothing else has already flagged the
    point.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0], [1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]]),
    ...     ["a", "b"],
    ... )
    >>> OrAggregator().aggregate(ts)["anomaly"].tolist()
    [1.0, 1.0, nan, 1.0]
    """

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        anomalous, unknown = _states(ts.values)
        combined = np.where(
            anomalous.any(axis=1), 1.0, np.where(unknown.any(axis=1), np.nan, 0.0)
        )
        return ts.wrap(combined, ["anomaly"])


class AndAggregator(BaseAggregator):
    """Flag a point only where every input agrees.

    The intersection of the inputs: use it to demand corroboration, so that a
    point counts only when several independent tests object to it. One definite
    "normal" settles the row; short of that, an unknown input leaves the answer
    unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0], [1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]]),
    ...     ["a", "b"],
    ... )
    >>> AndAggregator().aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 0.0, nan]
    """

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        anomalous, unknown = _states(ts.values)
        normal = ~anomalous & ~unknown
        combined = np.where(
            normal.any(axis=1), 0.0, np.where(unknown.any(axis=1), np.nan, 1.0)
        )
        return ts.wrap(combined, ["anomaly"])


class VoteAggregator(BaseAggregator):
    """Flag a point when enough of the inputs do.

    The middle ground between :class:`OrAggregator` and :class:`AndAggregator`,
    and the natural way to ensemble many detectors of similar quality: one
    detector's false positive is outvoted, while a genuine anomaly that most of
    them see survives.

    Unknown inputs abstain rather than vote against, so a detector that cannot
    label the start of a series does not drag the fraction down there.

    Parameters
    ----------
    threshold
        Fraction of the *known* inputs that must flag a point, from 0 to 1
        inclusive. The default of 0.5 is a simple majority, counting a tie as
        anomalous. At 1 every known input must agree; at 0 every row with any
        known label is flagged.

    Raises
    ------
    ValueError
        ``threshold`` lies outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, np.nan, np.nan]]),
    ...     ["a", "b", "c"],
    ... )
    >>> VoteAggregator(threshold=0.5).aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 1.0]
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        if not 0.0 <= self.threshold <= 1.0:
            msg = (
                f"threshold must be a fraction in [0, 1], got {self.threshold}. "
                f"It is the share of inputs that must agree, not a count."
            )
            raise ValueError(msg)

        anomalous, unknown = _states(ts.values)
        known = (~unknown).sum(axis=1)
        votes = anomalous.sum(axis=1)
        # A row where every input abstains divides zero by zero; the result is
        # discarded below, so the warning it would raise is not worth printing.
        with np.errstate(invalid="ignore", divide="ignore"):
            share = votes / known
        combined = np.where(known > 0, (share >= self.threshold).astype(float), np.nan)
        return ts.wrap(combined, ["anomaly"])


class CustomizedAggregator(BaseAggregator):
    """Wrap a user function into an aggregator.

    The contract is numpy in, numpy out: ``aggregate_func(labels, **params)``
    receives a ``float64`` array of shape ``(n_rows, n_inputs)`` whose columns
    are the inputs in the order they were passed, with unknown labels as
    ``NaN``, and must return an array of shape ``(n_rows,)`` — or
    ``(n_rows, 1)``, which is flattened. Returning ``NaN`` for a row means the
    combination is unknown there.

    Parameters
    ----------
    aggregate_func
        The combining function, as described above.
    aggregate_func_params
        Extra keyword arguments for ``aggregate_func``.

    Raises
    ------
    ValueError
        The function returned something other than one value per row.

    Examples
    --------
    A weighted vote, trusting the first input twice as much as the second:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    ...     ["a", "b"],
    ... )
    >>> weighted = CustomizedAggregator(
    ...     aggregate_func=lambda labels, w: (labels @ w >= 0.5).astype(float),
    ...     aggregate_func_params={"w": np.array([0.67, 0.33])},
    ... )
    >>> weighted.aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 0.0]
    """

    def __init__(
        self,
        aggregate_func: Callable[..., Any],
        aggregate_func_params: dict[str, Any] | None = None,
    ) -> None:
        self.aggregate_func = aggregate_func
        self.aggregate_func_params = aggregate_func_params

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        params = self.aggregate_func_params or {}
        result = np.asarray(self.aggregate_func(ts.values, **params), dtype=np.float64)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        if result.shape != (ts.n_rows,):
            msg = (
                f"aggregate_func returned an array of shape {result.shape}, but "
                f"combining {ts.n_columns} inputs over {ts.n_rows} rows must "
                f"give shape ({ts.n_rows},)."
            )
            raise ValueError(msg)
        return ts.wrap(result, ["anomaly"])


def _states(
    labels: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split labels into "anomalous" and "unknown" masks.

    Returns
    -------
    tuple of numpy.ndarray
        Two boolean arrays shaped like ``labels``. A cell is anomalous when it
        is known and non-zero; the two masks are disjoint, and a cell in neither
        is a definite "normal".
    """
    unknown = np.isnan(labels)
    anomalous = ~unknown & (labels != 0.0)
    return anomalous, unknown
