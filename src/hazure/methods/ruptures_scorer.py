"""Segmentation by the search strategies the ruptures package provides."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "RupturesScorer",
]


from hazure.methods.breakpoint_scorer import _BreakpointScorer
from hazure.methods.pelt_scorer import _default_penalty

#: Search strategies :class:`RupturesScorer` can dispatch to.
_RUPTURES_MODELS: Final = ("binseg", "window", "dynp", "bottomup")


class RupturesScorer(_BreakpointScorer):
    """Segment the series with one of the ``ruptures`` search strategies.

    An adapter, for the searches the built-in :class:`PeltScorer` does not
    provide. The one that earns its keep is ``n_bkps``: binary segmentation and
    dynamic programming can both be asked for *exactly* k breakpoints, which is
    the natural way to state the problem when the number of regimes is known and
    a penalty would have to be tuned backwards into it.

    Parameters
    ----------
    model
        Search strategy: ``"binseg"`` (greedy binary segmentation, fast),
        ``"window"`` (sliding two-window discrepancy), ``"dynp"`` (exhaustive
        dynamic programming, exact but expensive and requiring ``n_bkps``), or
        ``"bottomup"`` (greedy merging of an over-fine partition).
    cost
        A cost model name ``ruptures`` understands, such as ``"l1"``, ``"l2"``,
        ``"rbf"`` or ``"normal"``. Unlike :class:`PeltScorer`, this is not limited
        to the two costs hazure implements itself.
    penalty
        Cost of admitting one more segment. Ignored when ``n_bkps`` is given.
        None derives the same BIC-style value :class:`PeltScorer` uses.
    n_bkps
        Ask for exactly this many breakpoints instead of penalising their number.

    Attributes
    ----------
    breakpoints_ : numpy.ndarray
        Positions at which a new segment starts.
    penalty_ : float
        The penalty used, or NaN when ``n_bkps`` was given.

    Raises
    ------
    ValueError
        ``model`` is not one of the four strategies, or ``n_bkps`` is not
        positive.
    ImportError
        ``ruptures`` is not installed.

    Notes
    -----
    ``ruptures`` requires Python below 3.14, so this adapter is unavailable on
    newer interpreters. :class:`PeltScorer` is the portable option: it needs
    nothing beyond numpy, runs everywhere hazure runs, and solves the penalised
    problem exactly.

    Missing observations are filled by linear interpolation before the search,
    because the cost models take a dense matrix. Where that is unacceptable, use
    :class:`PeltScorer`, which skips them.

    Examples
    --------
    >>> from hazure.methods import RupturesScorer
    >>> RupturesScorer(model="dynp", n_bkps=1)  # doctest: +SKIP
    RupturesScorer(model='dynp', cost='l2', penalty=None, n_bkps=1)
    """

    def __init__(
        self,
        model: str = "binseg",
        cost: str = "l2",
        penalty: float | None = None,
        n_bkps: int | None = None,
    ) -> None:
        _check_ruptures(model, penalty, n_bkps)
        self.model = model
        self.cost = cost
        self.penalty = penalty
        self.n_bkps = n_bkps

    def _segment(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        _check_ruptures(self.model, self.penalty, self.n_bkps)
        algorithm = _ruptures_algorithm(self.model, self.cost)

        missing = np.isnan(values)
        dense = values
        if bool(missing.any()):
            if bool(missing.all()):
                self.penalty_ = math.nan
                return np.zeros(0, dtype=np.int64)
            positions = np.arange(values.shape[0], dtype=np.float64)
            dense = values.copy()
            dense[missing] = np.interp(
                positions[missing], positions[~missing], values[~missing]
            )

        fitted = algorithm.fit(dense.reshape(-1, 1))
        if self.n_bkps is not None:
            self.penalty_ = math.nan
            found = fitted.predict(n_bkps=int(self.n_bkps))
        else:
            self.penalty_ = (
                _default_penalty(values, "l1" if self.cost == "l1" else "l2")
                if self.penalty is None
                else float(self.penalty)
            )
            found = fitted.predict(pen=self.penalty_)
        # ruptures reports the end of each segment, so the last entry is the
        # length of the series and the rest are the starts of the segments after
        # the first — which is hazure's convention already.
        return np.asarray(found[:-1], dtype=np.int64)


def _check_ruptures(model: str, penalty: float | None, n_bkps: int | None) -> None:
    """Reject adapter parameters before anything is imported.

    Parameters
    ----------
    model
        Search strategy name.
    penalty
        Cost of one more segment, or None.
    n_bkps
        Requested number of breakpoints, or None.

    Raises
    ------
    ValueError
        The model is unknown, the penalty is negative, or ``n_bkps`` is below 1.
    """
    if model not in _RUPTURES_MODELS:
        msg = f"model={model!r} is not one of {list(_RUPTURES_MODELS)}."
        raise ValueError(msg)
    if penalty is not None and penalty < 0:
        msg = f"penalty={penalty} must not be negative."
        raise ValueError(msg)
    if n_bkps is not None and n_bkps < 1:
        msg = f"n_bkps must be at least 1 when given, got {n_bkps}."
        raise ValueError(msg)


def _ruptures_algorithm(model: str, cost: str) -> Any:
    """Build an unfitted ``ruptures`` estimator.

    Parameters
    ----------
    model
        One of :data:`_RUPTURES_MODELS`.
    cost
        Cost model name to hand to ``ruptures``.

    Returns
    -------
    Any
        The estimator, ready for ``fit``.

    Raises
    ------
    ImportError
        ``ruptures`` is not installed, or the interpreter is too new for it.
    """
    try:
        import ruptures
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            "RupturesScorer needs the ruptures package. Install it with "
            "`pip install hazure[cpd]`, which requires Python below 3.14; on "
            "newer interpreters use PeltScorer, which needs only numpy."
        )
        raise ImportError(msg) from exc

    builders = {
        "binseg": ruptures.Binseg,
        "window": ruptures.Window,
        "dynp": ruptures.Dynp,
        "bottomup": ruptures.BottomUp,
    }
    return builders[model](model=cost)
