"""Detectors for the moment the series became a different series.

A change point is not a point-in-time anomaly: it is the boundary between two
regimes, found by partitioning the whole series. There is deliberately no factor
to tune — the search's penalty has already decided which changes are large enough
to be worth a segment, and second-guessing that with a rule on the score would be
answering the same question twice with less information.

Like every change-point detector, these report the *moment of change* and then go
quiet, not the stretch that follows it. Evaluating them against interval-shaped
ground truth shows near-zero recall while they are working perfectly; reduce the
truth to change points and allow a tolerance with
:func:`~hazure.events.expand_events`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from hazure._core import Detector
from hazure.scorers import PeltScorer, RupturesScorer
from hazure.thresholds import FixedThreshold

if TYPE_CHECKING:
    from hazure.scorers import Cost

__all__ = [
    "pelt",
    "ruptures",
]

#: The scores are change sizes at breakpoints and zero elsewhere, so anything
#: above zero is a breakpoint the search chose.
_BREAKPOINT: Final = 0.0


def pelt(
    penalty: float | None = None, cost: Cost = "l2", min_size: int = 2, jump: int = 1
) -> Detector:
    """Flag the points at which the series changed regime.

    :class:`~hazure.scorers.PeltScorer` with a threshold that passes its non-zero
    scores through. To report fewer changes, raise ``penalty``.

    Parameters
    ----------
    penalty
        Cost of admitting one more segment. None derives a BIC-style value from
        the data.
    cost
        ``"l2"`` for squared deviations from the segment mean, ``"l1"`` for
        absolute deviations from its median.
    min_size
        Shortest segment allowed, in observations.
    jump
        Consider only breakpoints at multiples of this many observations.

    Returns
    -------
    Detector
        The PELT change sizes against ``FixedThreshold(high=0.0)``.

    Raises
    ------
    ValueError
        ``cost`` is unknown, or a size is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> values = np.concatenate([rng.normal(size=60), rng.normal(loc=10.0, size=60)])
    >>> time = np.arange("2024-01-01", "2024-04-30", dtype="datetime64[D]")
    >>> labels = pelt().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([60])
    """
    scorer = PeltScorer(penalty=penalty, cost=cost, min_size=min_size, jump=jump)
    return Detector(scorer, FixedThreshold(high=_BREAKPOINT))


def ruptures(
    model: str = "binseg",
    cost: str = "l2",
    penalty: float | None = None,
    n_bkps: int | None = None,
) -> Detector:
    """Flag the points at which the series changed regime, via ``ruptures``.

    :class:`~hazure.scorers.RupturesScorer` with a threshold that passes its
    non-zero scores through, exactly as :func:`pelt` does for the built-in search.
    Raise ``penalty``, or set ``n_bkps``, to report fewer.

    Reach for this over :func:`pelt` for one of two reasons — a cost model hazure
    does not implement (``"rbf"`` and ``"normal"`` detect changes in distribution
    rather than only in mean), or a **known number** of changes, which ``n_bkps``
    states directly instead of tuning a penalty backwards into it.

    Parameters
    ----------
    model
        Search strategy: ``"binseg"``, ``"window"``, ``"dynp"`` or ``"bottomup"``.
    cost
        A cost model name ``ruptures`` understands, such as ``"l1"``, ``"l2"``,
        ``"rbf"`` or ``"normal"``.
    penalty
        Cost of admitting one more segment. Ignored when ``n_bkps`` is given.
        None derives a BIC-style value from the data.
    n_bkps
        Ask for exactly this many breakpoints instead of penalising their number.

    Returns
    -------
    Detector
        The change sizes against ``FixedThreshold(high=0.0)``.

    Raises
    ------
    ValueError
        ``model`` is not one of the four strategies, or ``n_bkps`` is not
        positive.
    ImportError
        ``ruptures`` is not installed, when fitting.

    Notes
    -----
    ``ruptures`` requires Python below 3.14, so this is unavailable on newer
    interpreters; :func:`pelt` needs nothing beyond numpy.

    Examples
    --------
    >>> ruptures(model="dynp", n_bkps=1).scorer
    RupturesScorer(model='dynp', n_bkps=1)
    """
    scorer = RupturesScorer(model=model, cost=cost, penalty=penalty, n_bkps=n_bkps)
    return Detector(scorer, FixedThreshold(high=_BREAKPOINT))
