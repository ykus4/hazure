"""A signed score over a whole frame, reported as one verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hazure.detection.multivariate_score import _as_frame_label
from hazure.detection.signed_score import SignedScoreDetector

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "MultivariateSignedScoreDetector",
]


class MultivariateSignedScoreDetector(SignedScoreDetector):
    """A signed pairing whose scorer needs every column at once.

    Combines the direction filter of :class:`SignedScoreDetector` with the
    whole-frame view of :class:`MultivariateScoreDetector`.
    """

    multivariate: ClassVar[bool] = True

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return _as_frame_label(super()._compute(ts))
