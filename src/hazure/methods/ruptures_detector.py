"""Flagging the breakpoints a ruptures search strategy finds."""

from __future__ import annotations

from hazure.detection import ScoreDetector
from hazure.methods.ruptures_scorer import RupturesScorer
from hazure.thresholds import FixedThreshold

__all__ = [
    "RupturesDetector",
]


class RupturesDetector(ScoreDetector):
    """Flag the points at which the series changed regime, via ``ruptures``.

    :class:`RupturesScorer` with a threshold that passes its non-zero scores
    through, exactly as :class:`PeltDetector` does for the built-in search. There
    is no factor to tune, because the search has already decided which changes are
    worth a segment: raise ``penalty``, or set ``n_bkps``, to report fewer.

    Reach for this over :class:`PeltDetector` for one of two reasons — a cost model
    hazure does not implement (``"rbf"`` and ``"normal"`` detect changes in
    distribution rather than only in mean), or a **known number** of changes,
    which ``n_bkps`` states directly instead of tuning a penalty backwards into it.

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

    Raises
    ------
    ValueError
        ``model`` is not one of the four strategies, or ``n_bkps`` is not
        positive.
    ImportError
        ``ruptures`` is not installed.

    Notes
    -----
    Like every change-point detector here, this reports the *moment of change* and
    then goes quiet — not the stretch that follows it. Evaluating it against
    interval-shaped ground truth shows near-zero recall while it is working
    perfectly; reduce the truth to change points and allow a tolerance with
    :func:`~hazure.events.expand_events`.

    ``ruptures`` requires Python below 3.14, so this is unavailable on newer
    interpreters; :class:`PeltDetector` needs nothing beyond numpy.

    Examples
    --------
    >>> from hazure.methods import RupturesDetector
    >>> RupturesDetector(model="dynp", n_bkps=1)
    RupturesDetector(model='dynp', n_bkps=1)
    """

    def __init__(
        self,
        model: str = "binseg",
        cost: str = "l2",
        penalty: float | None = None,
        n_bkps: int | None = None,
    ) -> None:
        self.model = model
        self.cost = cost
        self.penalty = penalty
        self.n_bkps = n_bkps
        self._build()

    def _build(self) -> None:
        """Rebuild the scorer and the threshold from the current parameters."""
        self.scorer = RupturesScorer(
            model=self.model,
            cost=self.cost,
            penalty=self.penalty,
            n_bkps=self.n_bkps,
        )
        self.threshold = FixedThreshold(high=0.0)
