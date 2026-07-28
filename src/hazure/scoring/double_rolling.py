"""How much the window before each point disagrees with the window from it.

Configured three ways, this is where the spike, level-shift and
volatility-shift detectors come from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from hazure.features import (
    DoubleRollingAggregate,
)
from hazure.scoring.transformer_scorer import TransformerScorer

if TYPE_CHECKING:
    from hazure import BaseTransformer
    from hazure._core.window import Window

__all__ = [
    "DoubleRollingScorer",
]


#: How to compare the two windows of :class:`DoubleRollingScorer`.
Diff = Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"]


class DoubleRollingScorer(TransformerScorer):
    """Score each point by how much the series changes across it.

    The window before each point is summarised, the window from it onwards is
    summarised, and the two are compared. One scorer covers three phenomena, and
    only the settings differ:

    * **spike** — a long left window, a right window of 1, so a single blip is
      measured against a stable notion of recent normal;
    * **level shift** — two windows of equal length, long enough that both sides
      are stable, so a persistent change registers and a lone spike does not;
    * **volatility shift** — the same symmetric windows with a dispersion
      statistic (``agg="std"``) and a relative comparison
      (``diff="rel_diff"``), because a doubling of noise matters equally whether
      the series is quiet or loud.

    Parameters
    ----------
    window
        One spec for both sides, or ``(left, right)``.
    agg
        One statistic for both sides, or ``(left, right)``. The median resists
        the very outliers being detected.
    diff
        How to compare the two sides: ``"l1"`` or ``"l2"`` for the unsigned
        magnitude, ``"diff"`` for the signed ``right - left``, ``"rel_diff"`` for
        that divided by the left value, ``"abs_rel_diff"`` for its magnitude.
    min_periods
        Minimum non-missing observations per side, or ``(left, right)``.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Examples
    --------
    A step of 5 registers at the step, and the score is unavailable within a
    window of each end:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    >>> DoubleRollingScorer(window=2, diff="diff").score(ts).values.ravel()
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        agg: str | tuple[str, str] = "median",
        diff: Diff = "l1",
        min_periods: int | tuple[int | None, int | None] | None = None,
        q: float | None = None,
    ) -> None:
        self.window = window
        self.agg = agg
        self.diff = diff
        self.min_periods = min_periods
        self.q = q

    def _new_transformer(self) -> BaseTransformer:
        return DoubleRollingAggregate(
            window=self.window,
            agg=self.agg,
            agg_params=None if self.q is None else {"q": self.q},
            min_periods=self.min_periods,
            diff=self.diff,
        )
