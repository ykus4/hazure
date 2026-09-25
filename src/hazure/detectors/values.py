"""Detectors for a value outside its usual range, judged on its own.

No context is needed to call these values odd, so there is no scorer: the series
is its own score, and the choice is only of where to draw the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure._core import Detector
from hazure.thresholds import (
    EsdThreshold,
    FixedThreshold,
    IqrThreshold,
    QuantileThreshold,
)

if TYPE_CHECKING:
    from hazure.thresholds import FactorSpec

__all__ = [
    "esd",
    "iqr",
    "limits",
    "quantile",
]


def limits(low: float | None = None, high: float | None = None) -> Detector:
    """Flag values outside a range the caller supplies.

    The simplest possible detector, and the only one that learns nothing: use it
    when the acceptable range is known in advance — a pressure limit, a
    service-level objective.

    Parameters
    ----------
    low
        Values below this are anomalous. None leaves the lower side unbounded.
    high
        Values above this are anomalous. None leaves the upper side unbounded.

    Returns
    -------
    Detector
        ``Detector(None, FixedThreshold(low, high))``, usable without fitting.

    Raises
    ------
    ValueError
        Both bounds are None, which would make the detector inert.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [20.0, 21.0, 45.0, 19.0, -5.0])
    >>> limits(low=0.0, high=40.0).detect(ts).values.ravel()
    array([0., 0., 1., 0., 1.])
    """
    return Detector(None, FixedThreshold(low=low, high=high))


def iqr(factor: FactorSpec = 3.0) -> Detector:
    """Flag values far outside the training inter-quartile range.

    The box-plot rule, and a sound default when nothing is known about the
    distribution: because quartiles ignore the tails, the outliers being looked
    for do not widen the range that is supposed to exclude them.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded.

    Returns
    -------
    Detector
        ``Detector(None, IqrThreshold(factor))``.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([10.0, 11, 12, 11, 10, 12, 11, 10, 11, 60])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> iqr().fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """
    return Detector(None, IqrThreshold(factor=factor))


def quantile(low: float | None = None, high: float | None = None) -> Detector:
    """Flag values in the tails of the training distribution.

    Makes no assumption about the shape of that distribution, only about how much
    of it is acceptable: ``high=0.99`` means "the top one per cent of what we have
    seen is worth a look".

    Parameters
    ----------
    low
        Lower quantile in ``[0, 1]``. None leaves the lower side unbounded.
    high
        Upper quantile in ``[0, 1]``. None leaves the upper side unbounded.

    Returns
    -------
    Detector
        ``Detector(None, QuantileThreshold(low, high))``.

    Raises
    ------
    ValueError
        Both quantiles are None, or one falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 90])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> quantile(high=0.9).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """
    return Detector(None, QuantileThreshold(low=low, high=high))


def esd(alpha: float = 0.05) -> Detector:
    """Flag values by the generalised extreme Studentized deviate test.

    Sets the line by a significance level rather than by a factor, which is
    useful when a false-positive rate is easier to justify than a multiple of a
    spread. Assumes the values are approximately normal; where that is doubtful,
    :func:`iqr` asks less of the data.

    Parameters
    ----------
    alpha
        Significance level, in ``(0, 1)``.

    Returns
    -------
    Detector
        ``Detector(None, EsdThreshold(alpha))``.

    Raises
    ------
    ValueError
        ``alpha`` is not in ``(0, 1)``.
    ImportError
        SciPy is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(1)
    >>> values = rng.normal(loc=20.0, size=60)
    >>> values[42] = 30.0
    >>> time = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    >>> labels = esd().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([42])
    """
    return Detector(None, EsdThreshold(alpha=alpha))
