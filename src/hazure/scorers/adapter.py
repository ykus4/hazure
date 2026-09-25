"""Using a transformer's output as the score itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure._core import Component, Scorer
from hazure._core.missing import complete_rows

if TYPE_CHECKING:
    from hazure._core import TimeSeries


__all__ = [
    "AsScorer",
]


class AsScorer(Scorer):
    """Assert that a transformer's output measures how unusual each point is.

    A scorer and a transformer differ in intent rather than in arithmetic: the
    transformer produces a series, and wrapping it here asserts that this
    particular series is a score — which is what makes it meaningful to put a
    threshold on. The arithmetic stays in one class, and the assertion is made
    where the model is assembled rather than by a second class repeating the
    transformer's parameters.

    Parameters
    ----------
    transformer
        The transformer whose output is the score: a rolling statistic, the
        difference between two adjacent windows, a seasonal residual, a
        reconstruction error. A :class:`~hazure.Pipeline` ending in a transformer
        works too.

    Raises
    ------
    TypeError
        ``transformer`` does not produce a series.

    Notes
    -----
    Training data with no fully observed row supports no model, and every score
    is then unknown. Answering "no idea" keeps an all-missing column from raising
    part-way through a pipeline, where the transformer on its own would refuse to
    fit.

    The wrapped transformer is a parameter, so its own parameters are reachable
    as ``transformer__name`` and its fitted attributes on ``.transformer`` — the
    learned profile of a seasonal residual is ``scorer.transformer.seasonal_``.

    Examples
    --------
    The difference between the window before each point and the point itself is
    a score for spikes:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.transformers import DoubleRollingAggregate
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    >>> scorer = AsScorer(DoubleRollingAggregate(window=2, diff="diff"))
    >>> scorer.score(ts).values.ravel()
    array([nan, nan, 2.5, 5. , 2.5, nan])

    A seasonal residual, with the learned profile on the transformer:

    >>> from hazure.transformers import SeasonalDecomposition
    >>> values = np.tile([0.0, 1.0, 0.0, -1.0], 4)
    >>> values[9] = 6.0
    >>> time = np.arange("2024-01-01", "2024-01-17", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> scorer = AsScorer(SeasonalDecomposition(period=4)).fit(ts)
    >>> scorer.transformer.period_
    4
    >>> int(np.argmax(scorer.score(ts).values))
    9
    """

    trained_: bool

    def __init__(self, transformer: Component) -> None:
        _check_transformer(transformer)
        self.transformer = transformer

    @property
    def is_multivariate(self) -> bool:
        """True when the transformer needs every column at once."""
        return self.transformer.is_multivariate

    @property
    def is_trainable(self) -> bool:
        """True when the transformer has something to learn."""
        return self.transformer.is_trainable

    def _learn(self, ts: TimeSeries) -> None:
        _check_transformer(self.transformer)
        self.trained_ = bool(complete_rows(ts.values).any())
        if self.trained_:
            self.transformer.fit(ts)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if self.is_trainable and not self.trained_:
            return ts.wrap(np.full(ts.n_rows, np.nan))
        return self.transformer.run(ts)


def _check_transformer(transformer: object) -> None:
    """Refuse anything that does not produce a series."""
    if not (isinstance(transformer, Component) and transformer.output_kind == "series"):
        msg = (
            f"AsScorer wraps a transformer, but {type(transformer).__name__} does "
            f"not produce a series."
        )
        raise TypeError(msg)
