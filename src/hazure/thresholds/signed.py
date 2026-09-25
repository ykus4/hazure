"""Judging a signed score by its size, and reporting only the wanted direction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, get_args

import numpy as np

from hazure._core import Component, Threshold
from hazure._core.validate import check_choice

if TYPE_CHECKING:
    from hazure._core import TimeSeries

__all__ = [
    "Side",
    "SignedThreshold",
]


#: Which direction of excursion counts as an anomaly.
Side = Literal["both", "positive", "negative"]


_SIDES: Final = get_args(Side)


class SignedThreshold(Threshold):
    """Threshold the magnitude of a signed score, then filter by its direction.

    Many scores carry a sign that says which way the series moved — a residual,
    the difference between the window before a point and the window after it.
    This asks the wrapped threshold "how big is too big" once, of ``|score|``, so
    the answer is symmetric; ``side`` then decides whether the direction of a
    large score is of interest. Detecting only the increases is therefore not a
    different algorithm, just a filter on the same one.

    Parameters
    ----------
    threshold
        The rule applied to the magnitude. A one-sided rule such as
        ``IqrThreshold(factor=(None, 3.0))`` is usual, since a magnitude has no
        interesting lower tail.
    side
        ``"both"``, ``"positive"`` for increases only, or ``"negative"`` for
        decreases only.

    Raises
    ------
    ValueError
        ``side`` is not one of the three directions.
    TypeError
        ``threshold`` does not produce labels.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.thresholds import IqrThreshold
    >>> scores = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 8.0, 0.1, -0.3, 0.2, -9.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]"), scores
    ... )
    >>> either = SignedThreshold(IqrThreshold(factor=(None, 3.0)))
    >>> np.flatnonzero(either.fit_apply(ts).values.ravel() == 1.0)
    array([5, 9])
    >>> rises = SignedThreshold(IqrThreshold(factor=(None, 3.0)), side="positive")
    >>> np.flatnonzero(rises.fit_apply(ts).values.ravel() == 1.0)
    array([5])
    """

    def __init__(self, threshold: Component, side: Side = "both") -> None:
        _check(threshold, side)
        self.threshold = threshold
        self.side = side

    @property
    def is_trainable(self) -> bool:
        """True when the wrapped threshold has something to learn."""
        return self.threshold.is_trainable

    def _learn(self, ts: TimeSeries) -> None:
        _check(self.threshold, self.side)
        self.threshold.fit(_magnitude(ts))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check(self.threshold, self.side)
        labels = self.threshold.run(_magnitude(ts))
        if self.side == "both":
            return labels
        values = labels.values
        wanted = ts.values > 0.0 if self.side == "positive" else ts.values < 0.0
        gated = np.where(wanted, values, 0.0)
        gated[np.isnan(values)] = np.nan
        return labels.wrap(gated, labels.columns)


def _magnitude(ts: TimeSeries) -> TimeSeries:
    """Return the series with every value replaced by its absolute value."""
    return ts.wrap(np.abs(ts.values), ts.columns)


def _check(threshold: object, side: object) -> None:
    """Reject a direction that is not one of the three, or a non-threshold."""
    check_choice(side, _SIDES, "side")
    if not (isinstance(threshold, Component) and threshold.output_kind == "labels"):
        msg = (
            f"SignedThreshold wraps a threshold, but {type(threshold).__name__} "
            f"does not produce labels."
        )
        raise TypeError(msg)
