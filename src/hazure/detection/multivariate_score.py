"""A score over a whole frame, reported as one verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from hazure.detection.score import ScoreDetector

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "MultivariateScoreDetector",
]


#: Column name every multivariate detector reports its labels under.
_LABEL_NAME: Final = "anomaly"


class MultivariateScoreDetector(ScoreDetector):
    """A pairing whose scorer needs every column at once.

    Fitting sees the whole frame rather than one column at a time, so the model
    can learn how the columns relate. The single label series is reported under
    the column name ``anomaly``, since it describes the frame as a whole and not
    any one of its columns.
    """

    multivariate: ClassVar[bool] = True

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return _as_frame_label(super()._compute(ts))


def _as_frame_label(labels: TimeSeries) -> TimeSeries:
    """Rename a whole-frame verdict, which belongs to no single column."""
    return labels.wrap(labels.values, [_LABEL_NAME])
