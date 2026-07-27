"""Presenting a transformer's output as an anomaly score."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseScorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import BaseTransformer, TimeSeries

__all__ = ["complete_rows"]


class TransformerScorer(BaseScorer):
    """Base for scorers whose score *is* a transformer's output series.

    A scorer and a transformer differ in intent rather than in arithmetic: the
    transformer produces a series, and the scorer asserts that this particular
    series measures how unusual each point is, which is what makes it meaningful
    to put a threshold on. Subclasses need only say which transformer, and how it
    is configured from the scorer's own parameters.

    Attributes
    ----------
    transformer_ : BaseTransformer
        The transformer doing the work, available for inspection. Present after
        :meth:`fit` for trainable scorers; untrainable ones build it per call so
        that a parameter change takes effect immediately.
    """

    transformer_: BaseTransformer

    #: False when training data held nothing a model could be fitted to.
    _trained: bool = True

    @abstractmethod
    def _new_transformer(self) -> BaseTransformer:
        """Return a fresh transformer configured from the current parameters.

        Returns
        -------
        BaseTransformer
            An unfitted transformer.
        """

    def _learn(self, ts: TimeSeries) -> None:
        self.transformer_ = self._new_transformer()
        # Training data with no fully observed row supports no model, and every
        # score is then unknown. Answering "no idea" keeps an all-missing column
        # from raising part-way through a pipeline.
        self._trained = bool(complete_rows(ts.values).any())
        if self._trained and self.transformer_.trainable:
            self.transformer_.fit(ts)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if self.trainable and not self._trained:
            return ts.wrap(np.full(ts.n_rows, np.nan))
        transformer = self.transformer_ if self.trainable else self._new_transformer()
        return transformer.run(ts)


def complete_rows(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark rows with no missing value, the only rows a row-wise model can use."""
    usable: NDArray[np.bool_] = np.asarray(
        ~np.isnan(values).any(axis=1), dtype=np.bool_
    )
    return usable
