"""Reading a label column as anomalous, normal or unknown."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _states(
    labels: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split labels into "anomalous" and "unknown" masks.

    Returns
    -------
    tuple of numpy.ndarray
        Two boolean arrays shaped like ``labels``. A cell is anomalous when it
        is known and non-zero; the two masks are disjoint, and a cell in neither
        is a definite "normal".
    """
    unknown = np.isnan(labels)
    anomalous = ~unknown & (labels != 0.0)
    return anomalous, unknown
