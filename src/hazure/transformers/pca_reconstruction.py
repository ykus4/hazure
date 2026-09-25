"""Each point as the retained components can express it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure.transformers.pca_base import _PcaBase

if TYPE_CHECKING:
    from hazure._core import TimeSeries


__all__ = [
    "PcaReconstruction",
]


class PcaReconstruction(_PcaBase):
    """Rebuild each point from the leading principal components only.

    The output keeps the input's column names: it is the input as the model
    believes it should have looked, with everything outside the ``k``-dimensional
    subspace discarded. Rows with any missing value are missing throughout.

    Parameters
    ----------
    k
        Number of principal components to keep.
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores, complete = self._project(ts)
        rebuilt = np.full_like(ts.values, np.nan)
        rebuilt[complete] = self._reconstruct(scores[complete])
        return ts.wrap(rebuilt, ts.columns)
