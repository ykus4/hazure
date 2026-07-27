"""The five component types every algorithm in hazure is one of.

    Scorer      TimeSeries -> continuous score   (how unusual is each point?)
    Threshold   score       -> binary labels     (where do we draw the line?)
    Detector    a Scorer and Threshold together, ready to use
    Aggregator  several label series -> one      (ensembling)
    Transformer TimeSeries -> TimeSeries         (feature engineering)

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
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from hazure._core.config import Configurable
from hazure._core.series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "BaseAggregator",
    "BaseDetector",
    "BaseScorer",
    "BaseThreshold",
    "BaseTransformer",
    "Component",
]

_S = TypeVar("_S", bound="Component")


class Component(Configurable, ABC):
    """Shared machinery for scorers, thresholds, transformers and detectors.

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

    # Declared at class level rather than set in ``__init__`` so that a subclass
    # defining its own constructor cannot break the component by forgetting to
    # call super(). Fitting rebinds these on the instance.
    _fitted: bool = False
    _feature_names: tuple[str, ...] | None = None
    #: Populated only when a univariate component was fitted on a frame.
    _column_models: dict[str, Component] | None = None

    # -- hooks for subclasses ----------------------------------------------

    def _learn(self, ts: TimeSeries) -> None:
        """Learn from one series, which is univariate unless ``multivariate``.

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
            Input, already validated. Univariate unless :attr:`multivariate`.

        Returns
        -------
        TimeSeries
            The result, on the same time axis as ``ts``.
        """

    # -- state --------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run, or if the component needs no fitting."""
        return self._fitted or not self.trainable

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

        if self.multivariate or ts.is_univariate:
            self._column_models = None
            self._learn(ts)
        else:
            self._column_models = {name: self.clone() for name in ts.columns}
            for name, model in self._column_models.items():
                model.fit(ts.select(name))

        self._fitted = True
        return self

    # -- application --------------------------------------------------------

    def run(self, ts: TimeSeries) -> TimeSeries:
        """Apply this component to a :class:`TimeSeries`, returning one.

        This is the composition entry point: pipelines and compound detectors
        chain components through ``run`` so intermediate results never make a
        round trip through a native dataframe.

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

        ts = self._check_columns(ts)
        if self.multivariate or ts.is_univariate:
            return self._compute(ts)
        return _combine(self._named_part(name, ts) for name in ts.columns)

    def _named_part(self, name: str, ts: TimeSeries) -> TimeSeries:
        """Apply the copy responsible for one column, and label its output."""
        model = self if self._column_models is None else self._column_models[name]
        result = model._compute(ts.select(name))
        if result.columns == (name,):
            return result
        # A component that widens one column into several (lagging, say) would
        # otherwise collide across columns, so qualify the names.
        return result.wrap(
            result.values, [f"{name}_{column}" for column in result.columns]
        )

    def _check_columns(self, ts: TimeSeries) -> TimeSeries:
        """Reconcile the input's columns with the ones training saw."""
        learned = self._feature_names
        if learned is None or not self._fitted:
            return ts

        if self._column_models is not None:
            missing = [c for c in ts.columns if c not in self._column_models]
            if missing:
                msg = (
                    f"{type(self).__name__} was fitted on {list(learned)} and "
                    f"has nothing trained for {missing}."
                )
                raise ValueError(msg)
            return ts

        if self.multivariate:
            missing = [c for c in learned if c not in ts.columns]
            if missing:
                msg = (
                    f"{type(self).__name__} was fitted on {list(learned)} but "
                    f"the input is missing {missing}."
                )
                raise ValueError(msg)
            # Reorder to the training layout; extra columns are dropped, since
            # the model has no coefficients for them.
            return ts.select(learned)
        return ts

    # -- native-facing plumbing --------------------------------------------

    _verb: ClassVar[str] = "run"

    def _emit(self, data: Any) -> Any:
        """Apply to native input and return native output of the same flavour."""
        ts = TimeSeries.from_any(data)
        return self.run(ts).to_native()

    def _fit_emit(self, data: Any) -> Any:
        """Fit on native input, then apply to it."""
        ts = TimeSeries.from_any(data)
        return self.fit(ts).run(ts).to_native()


class BaseScorer(Component):
    """Turns a series into a continuous anomaly score.

    A score is "how unusual is this point", on whatever scale the algorithm
    works in. Higher means more unusual. Scores are useful on their own for
    ranking, and become labels when passed through a
    :class:`BaseThreshold`.
    """

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


class BaseThreshold(Component):
    """Turns continuous scores into binary labels.

    Kept separate from scoring so one policy — a quantile, an inter-quartile
    range, a fixed cut-off — can be reused across every scorer, and swapped
    without touching the scorer.
    """

    _verb: ClassVar[str] = "apply"

    def apply(self, scores: Any) -> Any:
        """Label ``scores``.

        Parameters
        ----------
        scores
            Continuous scores, as produced by a :class:`BaseScorer`.

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
            Continuous scores, as produced by a :class:`BaseScorer`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        return self._fit_emit(scores)


class BaseTransformer(Component):
    """Turns a series into another series.

    Feature engineering: rolling aggregates, lagging, seasonal decomposition.
    Transformers sit upstream of scorers in a pipeline.
    """

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


class BaseDetector(Component):
    """Turns a series directly into binary anomaly labels.

    Most detectors are a :class:`BaseScorer` paired with a
    :class:`BaseThreshold`; this is the type that pairing presents itself as, so
    that the common case stays a single object with familiar parameters.
    """

    _verb: ClassVar[str] = "detect"

    def detect(self, data: Any) -> Any:
        """Detect anomalies in ``data``.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            Labels as 1.0 for anomalous, 0.0 for normal and NaN for unknown, in
            the same flavour as the input.
        """
        return self._emit(data)

    def fit_detect(self, data: Any) -> Any:
        """Fit on ``data`` and detect anomalies in it in one step.

        This is the usual entry point for unsupervised use, where the same
        series both defines "normal" and is searched for departures from it.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`TimeSeries`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        return self._fit_emit(data)


class BaseAggregator(Configurable, ABC):
    """Combines several label series into one.

    Aggregators sit outside the fit/apply hierarchy because they have nothing to
    learn and no single input: they take the outputs of several detectors and
    reduce them, which is how ensembling and multi-condition rules are built.
    """

    @abstractmethod
    def _combine(self, ts: TimeSeries) -> TimeSeries:
        """Reduce a frame of label columns to a single label column.

        Parameters
        ----------
        ts
            One column per input label series, aligned on a shared time axis.

        Returns
        -------
        TimeSeries
            A single-column series of labels.
        """

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
            combined = _combine(renamed)

        return self._combine(combined).to_native()


def _combine(parts: Iterable[TimeSeries]) -> TimeSeries:
    """Join an iterable of series into one, aligning on the time axis."""
    materialised = list(parts)
    if not materialised:  # pragma: no cover - callers always pass at least one
        msg = "Nothing to combine."
        raise ValueError(msg)
    first, *rest = materialised
    return first.join(*rest)
