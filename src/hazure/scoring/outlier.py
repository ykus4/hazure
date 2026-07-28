"""The verdict of an outlier model of yours, on its own -1 convention."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseScorer
from hazure._core.series import complete_rows

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "OutlierScorer",
]


class OutlierScorer(BaseScorer):
    """Score each point with a time-independent outlier detection model.

    Treats every observation as a point in as many dimensions as there are
    columns and asks a general-purpose outlier model about it, ignoring the time
    axis entirely. The score is 1.0 for an outlier and 0.0 otherwise, so
    ``FixedThreshold(high=0.5)`` turns it into labels.

    Parameters
    ----------
    model
        An outlier model marking outliers with ``-1``. Two shapes are supported:

        * ``fit(X)`` plus ``predict(X)`` — the model learns the notion of normal
          once and then judges any later series against it;
        * ``fit_predict(X)`` alone — the model can only label the batch it is
          given, so it re-runs on each series and judges every batch against
          itself. Several well-known outlier models, including local outlier
          factor in its default mode, work only this way.

    Attributes
    ----------
    model_ : object
        The fitted outlier model.

    Notes
    -----
    The model is deep-copied at :meth:`fit` time, leaving the caller's object
    untouched.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> class FarFromCentre:
    ...     def fit(self, X):
    ...         self.centre_ = np.median(X, axis=0)
    ...         return self
    ...     def predict(self, X):
    ...         far = np.abs(X - self.centre_).sum(axis=1) > 5.0
    ...         return np.where(far, -1, 1)
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]"),
    ...     np.column_stack([[0.0, 1, 0, 1, 20], [0.0, 1, 1, 0, 20]]),
    ...     ["a", "b"],
    ... )
    >>> OutlierScorer(FarFromCentre()).fit_score(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """

    multivariate: ClassVar[bool] = True

    model_: Any

    def __init__(self, model: Any) -> None:
        self.model = model

    def _learn(self, ts: TimeSeries) -> None:
        self.model_ = copy.deepcopy(self.model)
        usable = complete_rows(ts.values)
        if hasattr(self.model_, "fit") and usable.any():
            self.model_.fit(ts.values[usable])

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = np.full(ts.n_rows, np.nan, dtype=np.float64)
        usable = complete_rows(ts.values)
        if usable.any():
            rows = ts.values[usable]
            judge = (
                self.model_.predict
                if hasattr(self.model_, "predict")
                else self.model_.fit_predict
            )
            scores[usable] = (np.asarray(judge(rows)).ravel() == -1).astype(np.float64)
        return ts.wrap(scores, ["outlier"])
