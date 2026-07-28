"""What a learned seasonal shape fails to explain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hazure.features import (
    SeasonalDecomposition,
)
from hazure.scoring.transformer_scorer import TransformerScorer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from hazure import BaseTransformer

__all__ = [
    "SeasonalResidualScorer",
]


class SeasonalResidualScorer(TransformerScorer):
    """Score each point by what a repeating pattern fails to explain.

    A daily or weekly cycle is normal behaviour, so it belongs in the model
    rather than in the anomalies. Classic additive decomposition removes it: the
    seasonal profile is the average shape of one cycle, optionally on top of a
    moving-average trend, and the residual is the signed remainder. The profile
    is learned once, so a later series is judged against the pattern that used
    to hold rather than against its own.

    Requires a regular time axis at :meth:`fit`, since a cycle length in
    observations is only meaningful if observations are evenly spaced. Later
    series may have gaps: the phase of each timestamp is recovered arithmetically
    from the training anchor.

    Parameters
    ----------
    period
        Length of a cycle in observations. When None it is detected from the
        autocorrelation of the training series.
    trend
        Estimate and remove a moving-average trend as well. Adds a NaN margin of
        half a period at each end, where the centred average has no window.

    Attributes
    ----------
    period_ : int
        Cycle length used, whether given or detected.
    seasonal_ : numpy.ndarray
        The learned profile, of length ``period_``, phase 0 being the first
        training observation.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([0.0, 1.0, 0.0, -1.0], 4)
    >>> values[9] = 6.0
    >>> time = np.arange("2024-01-01", "2024-01-17", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> scorer = SeasonalResidualScorer(period=4).fit(ts)
    >>> scorer.period_
    4
    >>> int(np.argmax(scorer.score(ts).values))
    9
    """

    def __init__(self, period: int | None = None, trend: bool = False) -> None:
        self.period = period
        self.trend = trend

    def _new_transformer(self) -> BaseTransformer:
        return SeasonalDecomposition(
            period=self.period, trend=self.trend, component="residual"
        )

    @property
    def period_(self) -> int:
        """Cycle length used, whether given or detected."""
        decomposition: Any = self.transformer_
        return int(decomposition.period_)

    @property
    def seasonal_(self) -> NDArray[np.float64]:
        """The learned seasonal profile, one value per phase."""
        decomposition: Any = self.transformer_
        profile: NDArray[np.float64] = decomposition.seasonal_
        return profile
