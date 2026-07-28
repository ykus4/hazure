"""Comparing the window before each point with the window from it.

A single rolling statistic says what the neighbourhood looks like. Two of
them, either side of a boundary, say whether the neighbourhood changed —
which is what turns a rolling aggregate into a change detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from hazure import BaseTransformer, double_rolling, parse_duration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure._core.window import Window

__all__ = [
    "DoubleRollingAggregate",
]


from hazure.features.spec import _is_sequence, _pair

Diff = Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"]


class DoubleRollingAggregate(BaseTransformer):
    """Compare the window before each point with the window after it.

    This is the primitive behind spike, level-shift and volatility-shift
    detection: summarise the recent past and the near future separately, then
    measure the gap. A large gap means the series changed character there.

    ``window``, ``agg``, ``agg_params`` and ``min_periods`` each accept a
    2-tuple to configure the two sides independently. Asymmetric settings are
    what separate the use cases — a long left window characterises "normal",
    a short right window catches a blip.

    Parameters
    ----------
    window
        One spec for both sides, or ``(left, right)``.
    agg
        One statistic for both sides, or ``(left, right)``. ``"std"``, ``"iqr"``
        or ``"idr"`` turn this into volatility-shift detection.
    agg_params
        Extra arguments for the aggregation, or ``(left, right)``. Only
        ``{"q": ...}`` for ``agg="quantile"`` is meaningful, and the two sides
        must ask for the same quantile.
    center
        When True (the default) the row sits on the boundary between the two
        windows, so a change is reported at the observation where it happens.
        When False the result is reported at the *last* observation of the right
        window, which is the earliest point at which a trailing-only detector
        could have known about it.
    min_periods
        Minimum non-missing observations per side, or ``(left, right)``.
    diff
        How to compare the two sides: ``"l1"`` or ``"l2"`` for the unsigned
        magnitude, ``"diff"`` for the signed ``right - left``, ``"rel_diff"``
        for that divided by the left value, ``"abs_rel_diff"`` for its
        magnitude.

    Examples
    --------
    A step change shows up as a peak at the step:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]"),
    ...     np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]),
    ... )
    >>> DoubleRollingAggregate(window=2, diff="diff").run(ts).values.ravel()
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        agg: str | tuple[str, str] = "mean",
        agg_params: dict[str, Any] | tuple[Any, Any] | None = None,
        center: bool = True,
        min_periods: int | tuple[int | None, int | None] | None = None,
        diff: Diff = "l1",
    ) -> None:
        self.window = window
        self.agg = agg
        self.agg_params = agg_params
        self.center = center
        self.min_periods = min_periods
        self.diff = diff

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        boundary = double_rolling(
            ts.values[:, 0],
            self.window,
            self.agg,
            time=ts.time,
            diff=self.diff,
            min_periods=self.min_periods,
            q=self._quantile(),
        )
        if self.center:
            return ts.wrap(boundary)
        windows: tuple[Window, Window] = _pair(self.window)
        return ts.wrap(_report_at_window_end(boundary, ts.time, windows[1]))

    def _quantile(self) -> float | None:
        """Resolve the single quantile both windows use, if either needs one."""
        aggs: tuple[str, str] = _pair(self.agg)
        params: tuple[dict[str, Any] | None, dict[str, Any] | None] = _pair(
            self.agg_params
        )
        wanted: list[float] = []
        for agg, side_params, side in zip(aggs, params, ("left", "right"), strict=True):
            if agg == "hist":
                msg = (
                    "agg='hist' returns a vector per window, which "
                    "DoubleRollingAggregate cannot difference. Use a scalar "
                    "aggregation such as 'mean', 'median' or 'std'."
                )
                raise ValueError(msg)
            if agg != "quantile":
                continue
            q = (side_params or {}).get("q")
            if q is None:
                msg = (
                    f"agg='quantile' on the {side} window needs a quantile: "
                    f"pass agg_params={{'q': 0.9}}."
                )
                raise ValueError(msg)
            if _is_sequence(q):
                msg = (
                    f"The {side} window was given several quantiles {q!r}. "
                    f"DoubleRollingAggregate differences one number per side; "
                    f"pass a single float."
                )
                raise ValueError(msg)
            wanted.append(float(q))

        if not wanted:
            return None
        if len(wanted) == 2 and wanted[0] != wanted[1]:
            msg = (
                f"The two windows ask for different quantiles "
                f"({wanted[0]} and {wanted[1]}), but one quantile is shared by "
                f"both. Use the same q on each side."
            )
            raise ValueError(msg)
        return wanted[0]


def _report_at_window_end(
    values: NDArray[np.float64], time: NDArray[np.int64], window: Window
) -> NDArray[np.float64]:
    """Move each boundary result to the last observation of its right window.

    The right window at row ``j`` runs from ``j`` forward, so its last member is
    the row a trailing-only view would first be able to report at.
    """
    n_rows = time.shape[0]
    if n_rows == 0:
        return values
    if isinstance(window, int):
        target = np.arange(n_rows) + window - 1
    else:
        edge = time + parse_duration(window)
        target = np.searchsorted(time, edge, side="left") - 1
    shifted = np.full(n_rows, np.nan, dtype=np.float64)
    # A boundary whose right window runs off the end of the series has no row to
    # be reported at, and is dropped rather than piled onto the last one. Where a
    # gap does make two windows end on the same row, the later one wins.
    reportable = target < n_rows
    shifted[target[reportable]] = values[reportable]
    return shifted
