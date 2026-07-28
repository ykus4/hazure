"""Coordinates of each point in the subspace the columns normally occupy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "PcaProjection",
]


from hazure.features.pca_base import _PcaBase


class PcaProjection(_PcaBase):
    """Project each point onto the leading principal components.

    Emits ``pc0`` .. ``pc{k-1}``, the coordinates of each time point in the
    subspace the training data occupies. Rows with any missing value are
    missing throughout.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Examples
    --------
    Two perfectly correlated columns vary along one direction only:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.array([0.0, 1.0, 2.0, 3.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.column_stack([base, 2.0 * base]),
    ...     ["a", "b"],
    ... )
    >>> PcaProjection(k=1).fit_transform(ts).values.ravel().round(6)
    array([-3.354102, -1.118034,  1.118034,  3.354102])
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores, _ = self._project(ts)
        return ts.wrap(scores, [f"pc{i}" for i in range(scores.shape[1])])
