"""The component types every algorithm in hazure is one of.

    Scorer       TimeSeries -> continuous score   (how unusual is each point?)
    Threshold    score       -> binary labels     (where do we draw the line?)
    Transformer  TimeSeries -> TimeSeries         (feature engineering)
    Aggregator   several columns -> one           (ensembling)

and :class:`~hazure.Detector`, which is a scorer and a threshold held together.

Asking "how unusual is this point" and asking "is that unusual enough to report"
are different questions, and hazure keeps them apart. One threshold policy is
then reusable across every scorer, a scorer can be swapped without revisiting the
policy, and a score is useful on its own for ranking — none of which works if the
two are welded into a single detector class.

Univariate and multivariate components share one hierarchy rather than being
split into parallel trees. A component declares :attr:`Component.multivariate`
and :attr:`Component.trainable`, and this module supplies the rest: input
validation, the fitted-state check, and the fan-out that lets a univariate
component handle a multi-column frame as one independently fitted copy per
column.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

from hazure._core.fanout import check_columns, fit_columns, join_all, run_columns
from hazure._core.persist import Persistent
from hazure._core.series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Aggregator",
    "Component",
    "OutputKind",
    "Scorer",
    "Threshold",
    "Transformer",
]

_S = TypeVar("_S", bound="Component")

#: What a component emits: continuous scores, binary labels, or a new series.
OutputKind = Literal["score", "labels", "series"]


class Component(Persistent, ABC):
    """Shared machinery for every component.

    Subclasses implement :meth:`_compute`, and :meth:`_learn` if they need
    training. Everything else — accepting any backend, validating the time axis,
    fanning out across columns, checking that training happened — is handled
    here.

    Attributes
    ----------
    multivariate
        True when the algorithm needs every column at once, as PCA
        reconstruction does. False means the algorithm is per-series, and a
        multi-column input is handled by fanning out.
    trainable
        False for algorithms with nothing to learn, such as a fixed threshold.
        Those may be used without calling :meth:`fit`.
    """

    multivariate: ClassVar[bool] = False
    trainable: ClassVar[bool] = True

    #: What :meth:`run` emits; see :attr:`output_kind`.
    _output: ClassVar[OutputKind]
    #: The verb that applies this component to native data, for messages.
    _verb: ClassVar[str] = "run"

    _persisted: ClassVar[tuple[str, ...]] = (
        "_fitted",
        "_feature_names",
        "_column_models",
    )

    # Declared at class level rather than set in ``__init__`` so that a subclass
    # defining its own constructor cannot break the component by forgetting to
    # call super(). Fitting rebinds these on the instance.
    _fitted: bool = False
    _feature_names: tuple[str, ...] | None = None
    #: Populated only when a univariate component was fitted on a frame.
    _column_models: dict[str, Component] | None = None

    # -- hooks for subclasses ----------------------------------------------

    def _learn(self, ts: TimeSeries) -> None:
        """Learn from one series, which is univariate unless multivariate.

        The default does nothing, which is correct for untrainable components.

        Parameters
        ----------
        ts
            Training data, already validated and narrowed.
        """

    @abstractmethod
    def _compute(self, ts: TimeSeries) -> TimeSeries:
        """Produce this component's output for one series.

        Parameters
        ----------
        ts
            Input, already validated. Univariate unless :attr:`is_multivariate`.

        Returns
        -------
        TimeSeries
            The result, on the same time axis as ``ts``.
        """

    # -- description ---------------------------------------------------------

    @property
    def is_multivariate(self) -> bool:
        """True when this instance needs every column at once.

        Usually the class-level :attr:`multivariate`. A component assembled from
        others — a detector, a pipeline — answers for its parts instead.
        """
        return type(self).multivariate

    @property
    def is_trainable(self) -> bool:
        """True when this instance has something to learn.

        Usually the class-level :attr:`trainable`. A component assembled from
        others is trainable when any of its parts is.
        """
        return type(self).trainable

    @property
    def output_kind(self) -> OutputKind:
        """What this component emits: ``"score"``, ``"labels"`` or ``"series"``."""
        return self._output

    # -- state --------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run, or if the component needs no fitting."""
        return self._fitted or not self.is_trainable

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Columns seen during :meth:`fit`, or None before fitting."""
        return self._feature_names

    # -- fitting ------------------------------------------------------------

    def fit(self: _S, data: Any) -> _S:
        """Train on ``data`` and return self, for chaining.

        A univariate component given a frame of *k* columns trains *k*
        independent copies of itself, one per column, so each learns its own
        normal range.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Component
            This component.
        """
        ts = TimeSeries.from_any(data)
        self._feature_names = ts.columns
        if self.is_multivariate or ts.is_univariate:
            self._column_models = None
            self._learn(ts)
        else:
            self._column_models = fit_columns(self, ts)
        self._fitted = True
        return self

    # -- application --------------------------------------------------------

    def run(self, ts: TimeSeries) -> TimeSeries:
        """Apply this component to a :class:`TimeSeries`, returning one.

        This is the composition entry point: pipelines and detectors chain
        components through ``run`` so intermediate results never make a round
        trip through a native dataframe.

        Parameters
        ----------
        ts
            Input series.

        Returns
        -------
        TimeSeries
            The component's output.

        Raises
        ------
        RuntimeError
            The component needs training and has not been trained.
        ValueError
            The input is missing a column that training used, or a multivariate
            component was handed different columns than it learned from.
        """
        if not self.fitted:
            msg = (
                f"{type(self).__name__} must be fitted before use. Call fit(), "
                f"or use fit_{self._verb}()."
            )
            raise RuntimeError(msg)
        ts = check_columns(self, ts)
        if self.is_multivariate or ts.is_univariate:
            return self._compute(ts)
        return run_columns(self, self._column_models, ts)

    # -- native-facing plumbing --------------------------------------------

    def _emit(self, data: Any) -> Any:
        """Apply to native input and return native output of the same flavour."""
        return self.run(TimeSeries.from_any(data)).to_native()

    def _fit_emit(self, data: Any) -> Any:
        """Fit on native input, then apply to it."""
        ts = TimeSeries.from_any(data)
        return self.fit(ts).run(ts).to_native()


class Scorer(Component):
    """Turns a series into a continuous anomaly score.

    A score is "how unusual is this point", on whatever scale the algorithm
    works in. Higher means more unusual — or, for a signed score, further from
    normal in the direction of its sign. Scores are useful on their own for
    ranking, and become labels when passed through a :class:`Threshold`.
    """

    _output: ClassVar[OutputKind] = "score"
    _verb: ClassVar[str] = "score"

    def score(self, data: Any) -> Any:
        """Score ``data``.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            Continuous scores, in the same flavour as the input.
        """
        return self._emit(data)

    def fit_score(self, data: Any) -> Any:
        """Fit on ``data`` and score it in one step.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            Continuous scores, in the same flavour as the input.
        """
        return self._fit_emit(data)


class Threshold(Component):
    """Turns continuous scores into binary labels.

    Kept separate from scoring so one policy — a quantile, an inter-quartile
    range, a fixed cut-off — can be reused across every scorer, and swapped
    without touching the scorer.
    """

    _output: ClassVar[OutputKind] = "labels"
    _verb: ClassVar[str] = "apply"

    def apply(self, scores: Any) -> Any:
        """Label ``scores``.

        Parameters
        ----------
        scores
            Continuous scores, as produced by a :class:`Scorer`.

        Returns
        -------
        Any
            Labels as 1.0 for anomalous, 0.0 for normal and NaN for unknown, in
            the same flavour as the input.
        """
        return self._emit(scores)

    def fit_apply(self, scores: Any) -> Any:
        """Fit on ``scores`` and label them in one step.

        Parameters
        ----------
        scores
            Continuous scores, as produced by a :class:`Scorer`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        return self._fit_emit(scores)


class Transformer(Component):
    """Turns a series into another series.

    Feature engineering: rolling aggregates, lagging, seasonal decomposition.
    Transformers sit upstream of scorers in a pipeline, and one whose output
    measures how unusual each point is can be used as a scorer through
    :class:`~hazure.scorers.AsScorer`.
    """

    _output: ClassVar[OutputKind] = "series"
    _verb: ClassVar[str] = "transform"

    def transform(self, data: Any) -> Any:
        """Transform ``data``.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            The transformed series, in the same flavour as the input.
        """
        return self._emit(data)

    def fit_transform(self, data: Any) -> Any:
        """Fit on ``data`` and transform it in one step.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            The transformed series, in the same flavour as the input.
        """
        return self._fit_emit(data)


class Aggregator(Component):
    """Combines the columns of a frame — several verdicts — into one.

    An aggregator is an ordinary component that happens to need every column at
    once and to have nothing to learn, so it runs anywhere a component does: at
    the end of a :class:`~hazure.Graph` that joins several detectors, or after a
    step in a :class:`~hazure.Pipeline` that widens one series into several.
    :meth:`aggregate` is the convenience for combining series you already hold.
    """

    multivariate: ClassVar[bool] = True
    trainable: ClassVar[bool] = False
    _output: ClassVar[OutputKind] = "labels"
    _verb: ClassVar[str] = "aggregate"

    def aggregate(self, *label_sets: Any, names: Iterable[str] | None = None) -> Any:
        """Combine label series.

        Parameters
        ----------
        *label_sets
            Two or more label series, or a single frame whose columns are the
            series to combine.
        names
            Names for the inputs, used to disambiguate identically named series.
            Defaults to ``input_0``, ``input_1``, ...

        Returns
        -------
        Any
            A single label series, in the same flavour as the first input.

        Raises
        ------
        ValueError
            Nothing was passed, or a single input has only one column.
        """
        if not label_sets:
            msg = "aggregate() needs at least one label series."
            raise ValueError(msg)

        parts = [TimeSeries.from_any(item) for item in label_sets]
        if len(parts) == 1:
            if parts[0].is_univariate:
                msg = (
                    "aggregate() needs several label series: pass them as "
                    "separate arguments, or as a frame with one column each."
                )
                raise ValueError(msg)
            combined = parts[0]
        else:
            labels = (
                list(names)
                if names is not None
                else [f"input_{i}" for i in range(len(parts))]
            )
            if len(labels) != len(parts):
                msg = f"Got {len(labels)} names for {len(parts)} label series."
                raise ValueError(msg)
            # Two detectors commonly emit series of the same name, which would
            # collide on join, so relabel single-column inputs.
            renamed = [
                part.wrap(part.values, [label]) if part.is_univariate else part
                for part, label in zip(parts, labels, strict=True)
            ]
            combined = join_all(renamed)

        return self.run(combined).to_native()
