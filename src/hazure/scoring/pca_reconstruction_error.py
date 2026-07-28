"""How far a point sits from the subspace the columns normally occupy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hazure.features import PcaReconstructionError
from hazure.scoring.transformer_scorer import TransformerScorer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from hazure import BaseTransformer

__all__ = [
    "PcaReconstructionErrorScorer",
]


class PcaReconstructionErrorScorer(TransformerScorer):
    """Score each point by how far it lies from the principal subspace.

    Correlated columns confine every observation to a low-dimensional subspace of
    the space they nominally span. Principal component analysis finds that
    subspace, and the squared distance from a point to it says how far the
    columns have stopped agreeing with one another — which is exactly the kind of
    anomaly that hides from every column individually.

    Parameters
    ----------
    k
        Number of principal components to keep. The score is the squared
        Euclidean distance to the best rank-``k`` reconstruction.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Column means of the training data.
    components_ : numpy.ndarray
        The ``k`` leading components, shape ``(k, n_columns)``.

    Examples
    --------
    Two columns on a line lie in a one-dimensional subspace, so a point knocked
    off that line stands out even though neither of its coordinates is extreme:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.arange(8.0)
    >>> partner = 2.0 * base + 1.0
    >>> partner[3] = 2.0 * base[3] - 4.0
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]"),
    ...     np.column_stack([base, partner]),
    ...     ["a", "b"],
    ... )
    >>> scores = PcaReconstructionErrorScorer(k=1).fit_score(ts)
    >>> int(np.argmax(scores.values))
    3
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def _new_transformer(self) -> BaseTransformer:
        return PcaReconstructionError(k=self.k)

    @property
    def mean_(self) -> NDArray[np.float64]:
        """Column means of the training data."""
        pca: Any = self.transformer_
        centre: NDArray[np.float64] = pca.mean_
        return centre

    @property
    def components_(self) -> NDArray[np.float64]:
        """The leading principal components, one per row."""
        pca: Any = self.transformer_
        basis: NDArray[np.float64] = pca.components_
        return basis
