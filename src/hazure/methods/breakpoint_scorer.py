"""The shared base for scorers that segment a series into regimes.

A change point is not a point-in-time anomaly, so the score is sparse: the
size of the level change at each breakpoint, and zero everywhere else.
"""

from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np

from hazure import BaseScorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "Cost",
]


#: Cost functions the built-in segmentation understands.
Cost = Literal["l1", "l2"]


class _BreakpointScorer(BaseScorer):
    """Shared machinery for scorers whose model of the series is a partition.

    Subclasses supply :meth:`_segment`, which returns the positions where a new
    segment starts. Everything else — the change magnitudes, and how a partition
    learned from one series is expressed as a score on another — is here.

    Attributes
    ----------
    breakpoints_ : numpy.ndarray
        Positions in the training series at which a new segment starts, in
        increasing order. Empty when the series is one regime throughout. The
        first observation is not a breakpoint: every series starts a segment.
    penalty_ : float
        The penalty actually used, whether given or derived. NaN when the search
        was told how many breakpoints to find instead.
    """

    breakpoints_: NDArray[np.int64]
    penalty_: float

    #: Cost function name, which decides what a segment's level means. Set by
    #: every subclass's constructor.
    cost: str

    #: Timestamps of the breakpoints, so a learned partition can be located in a
    #: series that is not the training one.
    _change_time: NDArray[np.int64]
    #: Size of the level change at each breakpoint.
    _change_size: NDArray[np.float64]

    @abstractmethod
    def _segment(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Partition a series.

        Parameters
        ----------
        values
            1-D array, possibly with missing observations.

        Returns
        -------
        numpy.ndarray
            Positions at which a new segment starts, in increasing order.
        """

    @property
    def _level(self) -> str:
        """The statistic summarising where a segment sits.

        Returns
        -------
        str
            ``"median"`` when the cost function is absolute deviation, otherwise
            ``"mean"`` — in both cases the value the segment's cost is measured
            from, so the reported change is the change the search acted on.
        """
        return "median" if self.cost == "l1" else "mean"

    def _learn(self, ts: TimeSeries) -> None:
        values = ts.values[:, 0]
        self.breakpoints_ = self._segment(values)
        self._change_time = ts.time[self.breakpoints_]
        self._change_size = _change_sizes(values, self.breakpoints_, self._level)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = np.zeros(ts.n_rows, dtype=np.float64)
        if ts.n_rows == 0 or self._change_time.size == 0:
            return ts.wrap(scores)
        # The partition was found on the training series, so the breakpoints are
        # located by *when* they happened rather than by position: a series on the
        # same time axis scores identically, and one on a different axis gets the
        # learned changes wherever those instants fall in it.
        position = np.clip(
            np.searchsorted(ts.time, self._change_time), 0, ts.n_rows - 1
        )
        matched = ts.time[position] == self._change_time
        scores[position[matched]] = self._change_size[matched]
        return ts.wrap(scores)


def _change_sizes(
    values: NDArray[np.float64], breakpoints: NDArray[np.int64], level: str
) -> NDArray[np.float64]:
    """Measure the level change across each breakpoint.

    Parameters
    ----------
    values
        The series.
    breakpoints
        Positions at which a new segment starts.
    level
        ``"mean"`` or ``"median"``, matching the cost function.

    Returns
    -------
    numpy.ndarray
        One non-negative magnitude per breakpoint. Zero where a segment held no
        observations at all and no change could be measured.
    """
    if breakpoints.size == 0:
        return np.zeros(0, dtype=np.float64)
    edges = [0, *breakpoints.tolist(), values.shape[0]]
    with warnings.catch_warnings():
        # An entirely missing segment has no level; the change across it is
        # reported as zero below rather than as NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        levels = np.asarray(
            [
                np.nanmedian(values[a:b])
                if level == "median"
                else np.nanmean(values[a:b])
                for a, b in itertools.pairwise(edges)
            ],
            dtype=np.float64,
        )
    change = np.abs(np.diff(levels))
    return np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)
