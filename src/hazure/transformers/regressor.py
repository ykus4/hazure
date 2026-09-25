"""The interface a regressor has to satisfy to be used here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


__all__ = [
    "Regressor",
]


class Regressor(Protocol):
    """The interface :class:`RegressionResidual` needs from a regressor.

    Deliberately the same two methods scikit-learn estimators expose, so one can
    be passed straight in, but structural: any object with a matching ``fit`` and
    ``predict`` works and no import is implied.
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], /) -> Any:
        """Learn from a design matrix and a target vector."""

    def predict(self, X: NDArray[np.float64], /) -> NDArray[np.float64]:
        """Predict the target for each row of a design matrix."""
