"""What a multi-period seasonal-trend decomposition cannot explain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hazure import BaseScorer
from hazure.methods.stl_residual_scorer import _IRREGULAR, _residual_score

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hazure import TimeSeries


__all__ = [
    "MstlResidualScorer",
]


class MstlResidualScorer(BaseScorer):
    """Score each point by the size of its residual after removing several cycles.

    MSTL applies STL once per period, longest cycle last, subtracting each
    seasonal component before fitting the next. A series sampled hourly usually
    has two rhythms — the hour of the day and the day of the week — and a
    single-period decomposition has to choose one and leave the other in the
    residual, where it dominates and drowns everything else. Removing both leaves
    a remainder that is genuinely just the remainder.

    Parameters
    ----------
    periods
        Cycle lengths in observations: one integer, or several. For hourly data,
        ``(24, 168)`` is the daily-and-weekly pair.
    robust
        Reweight the loess fits to discount outliers, in every STL pass.
    windows
        Seasonal smoother length per period, as an integer or one per period.
        None leaves them to ``statsmodels``.

    Raises
    ------
    ValueError
        The time axis is irregular, ``periods`` is empty, or a period is below 2.
    ImportError
        ``statsmodels`` is not installed.

    Notes
    -----
    Requires a regular time axis. Missing observations are interpolated before the
    decomposition and their scores set back to NaN, as for
    :class:`StlResidualScorer`.

    Each period needs two full cycles of data to be estimable, so the longest
    period governs how much history the scorer needs.

    Nothing is learned; :meth:`fit` is optional.

    References
    ----------
    .. [1] K. Bandara, R. J. Hyndman and C. Bergmeir, "MSTL: A Seasonal-Trend
       Decomposition Algorithm for Time Series with Multiple Seasonal Patterns",
       International Journal of Operational Research, 2021.

    Examples
    --------
    A daily and a weekly rhythm together, with one point breaking both:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> hours = np.arange(24 * 28)
    >>> values = (
    ...     10.0
    ...     + 3.0 * np.sin(hours * 2 * np.pi / 24)
    ...     + 5.0 * np.sin(hours * 2 * np.pi / 168)
    ... )
    >>> values[300] += 20.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> scorer = MstlResidualScorer(periods=(24, 168))
    >>> scores = scorer.score(TimeSeries.from_arrays(time, values))
    >>> int(np.nanargmax(scores.values.ravel()))
    300
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        periods: int | Sequence[int],
        robust: bool = True,
        windows: int | Sequence[int] | None = None,
    ) -> None:
        _check_periods(periods)
        self.periods = periods
        self.robust = robust
        self.windows = windows

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check_periods(self.periods)
        if ts.freq is None:
            msg = _IRREGULAR.format(name="MstlResidualScorer")
            raise ValueError(msg)
        options: dict[str, Any] = {
            "periods": self.periods,
            "stl_kwargs": {"robust": self.robust},
        }
        if self.windows is not None:
            options["windows"] = self.windows
        return _residual_score(ts, "MSTL", options)


def _check_periods(periods: int | Sequence[int]) -> None:
    """Reject cycle lengths a multi-seasonal decomposition cannot use.

    Parameters
    ----------
    periods
        One period or several.

    Raises
    ------
    ValueError
        No periods were given, or one is below 2 observations.
    """
    candidates = [periods] if isinstance(periods, int) else list(periods)
    if not candidates:
        msg = (
            "periods is empty; pass at least one cycle length, e.g. "
            "periods=24 for a daily cycle in hourly data."
        )
        raise ValueError(msg)
    for value in candidates:
        if value < 2:
            msg = f"Every period must be at least 2 observations, got {value}."
            raise ValueError(msg)
