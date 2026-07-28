"""Each point as the retained components can express it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "PcaReconstruction",
]


from hazure.features.pca_base import _PcaBase


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
        rebuilt[complete] = self.mean_ + scores[complete] @ self.components_
        return ts.wrap(rebuilt, ts.columns)
