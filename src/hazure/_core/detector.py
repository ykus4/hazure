"""A scorer and a threshold, fitted together."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

from hazure._core.component import Component, OutputKind

if TYPE_CHECKING:
    from hazure._core.series import TimeSeries

__all__ = ["Detector"]

#: Name of the verdict a multivariate detector emits, which belongs to no column.
_FRAME_LABEL: Final = "anomaly"


class Detector(Component):
    """A scorer and a threshold, applied in that order.

    This is the one detector class. Every ready-made detector in
    :mod:`hazure.detectors` is a function returning one, with the scorer and the
    threshold chosen for a particular kind of anomaly — so any detector can be
    pulled apart, printed, reconfigured through ``set_params`` or stored, the
    same way as one you assembled yourself.

    Fitting fits the scorer, then fits the threshold on the scores the fitted
    scorer produces for the training data — so the threshold learns the scale the
    scorer actually works on, which is the whole reason the two are fitted
    together rather than independently.

    Parameters
    ----------
    scorer
        Anything producing scores: a :class:`~hazure.Scorer`, or a
        :class:`~hazure.Pipeline` ending in one. None when the series is already
        its own score, as it is for a plain value range.
    threshold
        The rule that turns those scores into labels.

    Raises
    ------
    TypeError
        ``scorer`` does not produce scores, or ``threshold`` does not produce
        labels.

    Notes
    -----
    A detector needs every column at once exactly when its scorer does. It then
    emits a single verdict for the whole frame, named ``anomaly``. Otherwise it
    fans out like any univariate component: one independently fitted scorer and
    threshold per column.

    Examples
    --------
    Any scorer pairs with any threshold:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.scorers import DeviationScorer
    >>> from hazure.thresholds import MadThreshold
    >>> values = np.array([5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 40.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]"), values
    ... )
    >>> detector = Detector(DeviationScorer(), MadThreshold())
    >>> detector.fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 1.])
    >>> detector.scorer.center_
    6.0

    The parts stay reachable by name, for reconfiguring or for grid search:

    >>> detector.set_params(threshold__factor=100.0).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0.])
    """

    _output: ClassVar[OutputKind] = "labels"
    _verb: ClassVar[str] = "detect"

    def __init__(self, scorer: Component | None, threshold: Component) -> None:
        _check_parts(scorer, threshold)
        self.scorer = scorer
        self.threshold = threshold

    # -- description ---------------------------------------------------------

    @property
    def is_multivariate(self) -> bool:
        """True when the scorer needs every column at once."""
        first = self.threshold if self.scorer is None else self.scorer
        return first.is_multivariate

    @property
    def is_trainable(self) -> bool:
        """True when the scorer or the threshold has something to learn."""
        parts = (
            [self.threshold] if self.scorer is None else [self.scorer, self.threshold]
        )
        return any(part.is_trainable for part in parts)

    # -- the work --------------------------------------------------------------

    def _learn(self, ts: TimeSeries) -> None:
        _check_parts(self.scorer, self.threshold)
        scores = ts if self.scorer is None else self.scorer.fit(ts).run(ts)
        self.threshold.fit(scores)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = ts if self.scorer is None else self.scorer.run(ts)
        labels = self.threshold.run(scores)
        if self.is_multivariate and labels.is_univariate:
            return labels.wrap(labels.values, [_FRAME_LABEL])
        return labels

    # -- verbs ---------------------------------------------------------------

    def detect(self, data: Any) -> Any:
        """Detect anomalies in ``data``.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        Any
            Labels as 1.0 for anomalous, 0.0 for normal and NaN for unknown, in
            the same flavour as the input.
        """
        return self._emit(data)

    def fit_detect(self, data: Any) -> Any:
        """Fit on ``data`` and detect anomalies in it in one step.

        This is the usual entry point for unsupervised use, where the same
        series both defines "normal" and is searched for departures from it.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        return self._fit_emit(data)


def _check_parts(scorer: object, threshold: object) -> None:
    """Refuse parts that do not produce what their position needs."""
    if scorer is not None and not (
        isinstance(scorer, Component) and scorer.output_kind == "score"
    ):
        msg = (
            f"Detector scorer must produce scores, but {type(scorer).__name__} "
            f"does not. A transformer whose output is the score can be used as "
            f"one through hazure.scorers.AsScorer."
        )
        raise TypeError(msg)
    if not (isinstance(threshold, Component) and threshold.output_kind == "labels"):
        msg = (
            f"Detector threshold must produce labels, but "
            f"{type(threshold).__name__} does not."
        )
        raise TypeError(msg)
