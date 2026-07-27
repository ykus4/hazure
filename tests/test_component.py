"""Tests for the component hierarchy and parameter introspection.

Exercised through small concrete components rather than the real algorithms, so
a failure here points at the base machinery instead of at an algorithm.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from hazure._core import (
    BaseAggregator,
    BaseDetector,
    BaseScorer,
    BaseThreshold,
    BaseTransformer,
    Configurable,
    TimeSeries,
)
from tests.conftest import BACKENDS, make_native

# -- test doubles -----------------------------------------------------------


class Doubler(BaseTransformer):
    """Multiplies by a factor. Untrainable, so usable straight away."""

    trainable: ClassVar[bool] = False

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(ts.values * self.factor)


class DeviationScorer(BaseScorer):
    """Scores each point by its distance from the training mean."""

    def __init__(self, absolute: bool = True) -> None:
        self.absolute = absolute

    def _learn(self, ts: TimeSeries) -> None:
        self.centre_ = float(np.nanmean(ts.values))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        deviation = ts.values[:, 0] - self.centre_
        return ts.wrap(np.abs(deviation) if self.absolute else deviation)


class OverMean(BaseThreshold):
    """Flags scores above the training mean times a factor."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        self.cutoff_ = float(np.nanmean(ts.values)) * self.factor

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap((ts.values[:, 0] > self.cutoff_).astype(float))


class Pairing(BaseDetector):
    """A scorer and a threshold presented as one detector."""

    def __init__(self, scorer: BaseScorer, threshold: BaseThreshold) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def _learn(self, ts: TimeSeries) -> None:
        self.threshold.fit(self.scorer.fit(ts).run(ts))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return self.threshold.run(self.scorer.run(ts))


class Widener(BaseTransformer):
    """Turns one column into two, to exercise output naming under fan-out."""

    trainable: ClassVar[bool] = False

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        stacked = np.column_stack([ts.values[:, 0], ts.values[:, 0] * -1.0])
        return ts.wrap(stacked, ["up", "down"])


class Total(BaseTransformer):
    """Sums every column, so it genuinely needs them all at once."""

    multivariate: ClassVar[bool] = True
    trainable: ClassVar[bool] = False

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nansum(ts.values, axis=1), ["total"])


class AnyOf(BaseAggregator):
    """Flags a point when any input flagged it."""

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nanmax(ts.values, axis=1), ["anomaly"])


@pytest.fixture
def stepped() -> pd.Series:
    """A quiet series with a clear step change part-way through."""
    index = pd.date_range("2024-01-01", periods=20, freq="h", name="time")
    values = np.concatenate([np.zeros(15), np.full(5, 10.0)])
    return pd.Series(values, index=index, name="sensor")


# -- the layering -----------------------------------------------------------


def test_a_scorer_returns_continuous_values() -> None:
    """A scorer is usable on its own, with no threshold in sight."""
    index = pd.date_range("2024-01-01", periods=20, freq="h", name="time")
    values = np.arange(20.0)
    values[7] = 500.0
    noisy = pd.Series(values, index=index, name="sensor")

    scores = DeviationScorer().fit_score(noisy)

    assert scores.nunique() > 2
    assert scores.idxmax() == index[7]


def test_a_threshold_can_be_swapped_without_touching_the_scorer(
    stepped: pd.Series,
) -> None:
    scorer = DeviationScorer().fit(stepped)
    scores = scorer.run(TimeSeries.from_any(stepped)).to_native()

    lenient = OverMean(factor=3.0).fit_apply(scores)
    strict = OverMean(factor=0.5).fit_apply(scores)

    assert strict.sum() > lenient.sum()


def test_a_detector_pairs_a_scorer_with_a_threshold(stepped: pd.Series) -> None:
    flags = Pairing(DeviationScorer(), OverMean()).fit_detect(stepped)
    assert set(flags.unique()) <= {0.0, 1.0}
    assert flags.loc[stepped.index[15:]].eq(1.0).all()
    assert flags.loc[stepped.index[:15]].eq(0.0).all()


def test_an_aggregator_reduces_several_label_series() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    left = pd.Series([1.0, 0.0, 0.0, 0.0], index=index, name="flag")
    right = pd.Series([0.0, 0.0, 1.0, 0.0], index=index, name="flag")

    combined = AnyOf().aggregate(left, right)

    assert list(combined) == [1.0, 0.0, 1.0, 0.0]
    assert combined.name == "anomaly"


def test_an_aggregator_accepts_a_single_frame() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=index)
    assert list(AnyOf().aggregate(frame)["anomaly"]) == [1.0, 0.0, 1.0]


def test_an_aggregator_aligns_disjoint_time_axes() -> None:
    """Alignment is an explicit join on the time axis, not an implicit one."""
    left = pd.Series(
        [1.0, 0.0],
        index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"], name="time"),
        name="flag",
    )
    right = pd.Series(
        [0.0, 1.0],
        index=pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 02:00"], name="time"),
        name="flag",
    )
    combined = AnyOf().aggregate(left, right)
    assert len(combined) == 3
    assert list(combined) == [1.0, 0.0, 1.0]


def test_an_aggregator_needs_more_than_one_series() -> None:
    single = pd.Series(
        [1.0], index=pd.date_range("2024-01-01", periods=1, name="time"), name="flag"
    )
    with pytest.raises(ValueError, match="needs several label series"):
        AnyOf().aggregate(single)


def test_an_aggregator_needs_at_least_one_argument() -> None:
    with pytest.raises(ValueError, match="at least one label series"):
        AnyOf().aggregate()


def test_an_aggregator_rejects_a_name_count_mismatch() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    part = pd.Series([1.0, 0.0], index=index, name="flag")
    with pytest.raises(ValueError, match="names for 2 label series"):
        AnyOf().aggregate(part, part, names=["only_one"])


# -- fan-out across columns -------------------------------------------------


def test_a_univariate_component_fans_out_over_a_frame() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="h", name="time")
    frame = pd.DataFrame(
        {"small": np.arange(10.0), "large": np.arange(10.0) * 1000},
        index=index,
    )
    scores = DeviationScorer().fit_score(frame)
    assert list(scores.columns) == ["small", "large"]
    # Each column learned its own centre, so both peak at their own extremes.
    assert scores["large"].max() > scores["small"].max() * 100


def test_fan_out_fits_each_column_independently() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    frame = pd.DataFrame({"a": np.zeros(6), "b": np.full(6, 100.0)}, index=index)
    scorer = DeviationScorer().fit(frame)

    assert scorer._column_models is not None
    centres = {n: m.centre_ for n, m in scorer._column_models.items()}  # type: ignore[attr-defined]
    assert centres == {"a": 0.0, "b": 100.0}


def test_a_widening_component_qualifies_its_output_names() -> None:
    """Without prefixing, two columns of ``up``/``down`` would collide."""
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    frame = pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)}, index=index)
    result = Widener().transform(frame)
    assert list(result.columns) == ["a_up", "a_down", "b_up", "b_down"]


def test_a_widening_component_keeps_plain_names_for_one_column() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    result = Widener().transform(pd.Series(np.arange(4.0), index=index, name="a"))
    assert list(result.columns) == ["up", "down"]


def test_a_multivariate_component_sees_every_column_at_once() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}, index=index)
    assert list(Total().transform(frame)["total"]) == [11.0, 22.0, 33.0]


def test_a_multivariate_component_reorders_columns_to_the_training_layout() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    trained = Total().fit(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}, index=index))
    swapped = pd.DataFrame({"b": [2.0, 2.0], "a": [1.0, 1.0]}, index=index)
    assert trained.feature_names == ("a", "b")
    assert list(trained.transform(swapped)["total"]) == [3.0, 3.0]


def test_a_multivariate_component_rejects_a_missing_column() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    trained = Total().fit(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}, index=index))
    with pytest.raises(ValueError, match=r"missing \['b'\]"):
        trained.transform(pd.DataFrame({"a": [1.0, 1.0]}, index=index))


def test_a_fanned_out_component_rejects_an_unseen_column() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    trained = DeviationScorer().fit(
        pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)}, index=index)
    )
    with pytest.raises(ValueError, match=r"nothing trained for \['c'\]"):
        trained.score(pd.DataFrame({"c": np.arange(4.0)}, index=index))


# -- fitting contract -------------------------------------------------------


def test_a_trainable_component_must_be_fitted_first(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        DeviationScorer().score(stepped)


def test_the_error_names_the_shortcut_method(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="fit_detect"):
        Pairing(DeviationScorer(), OverMean()).detect(stepped)


def test_an_untrainable_component_works_without_fitting(stepped: pd.Series) -> None:
    assert Doubler().fitted
    assert list(Doubler(factor=3.0).transform(stepped)[:3]) == [0.0, 0.0, 0.0]


def test_fit_returns_self_for_chaining(stepped: pd.Series) -> None:
    scorer = DeviationScorer()
    assert scorer.fit(stepped) is scorer


def test_feature_names_records_what_training_saw() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, index=index)
    assert DeviationScorer().fit(frame).feature_names == ("x", "y")


def test_feature_names_is_none_before_fitting() -> None:
    assert DeviationScorer().feature_names is None


def test_refitting_replaces_the_previous_state() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    scorer = DeviationScorer()
    scorer.fit(pd.DataFrame({"a": np.zeros(4)}, index=index))
    scorer.fit(pd.DataFrame({"b": np.zeros(4)}, index=index))
    assert scorer.feature_names == ("b",)
    assert scorer.score(pd.DataFrame({"b": np.zeros(4)}, index=index)) is not None


# -- backend independence ---------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_component_returns_the_backend_it_was_given(backend: str) -> None:
    native = make_native(backend, np.arange(10.0))
    result = DeviationScorer().fit_score(native)
    assert type(result) is type(native)


def test_every_backend_produces_the_same_scores() -> None:
    values = np.concatenate([np.zeros(8), np.full(4, 5.0)])
    results = [
        TimeSeries.from_any(DeviationScorer().fit_score(make_native(b, values)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_allclose(other.values, results[0].values)


# -- parameter introspection ------------------------------------------------


def test_get_params_reads_the_constructor_signature() -> None:
    """No hand-maintained list, so it cannot drift from ``__init__``."""
    assert Doubler(factor=5.0).get_params() == {"factor": 5.0}


def test_set_params_updates_in_place_and_chains() -> None:
    component = Doubler()
    assert component.set_params(factor=7.0) is component
    assert component.factor == 7.0


def test_set_params_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="not parameters of Doubler"):
        Doubler().set_params(multiplier=2.0)


def test_clone_preserves_every_parameter_including_nested_ones() -> None:
    """Every setting survives a clone, including those of nested components."""
    original = Pairing(DeviationScorer(absolute=False), OverMean(factor=2.5))
    copy = original.clone()

    assert copy.scorer.absolute is False
    assert copy.threshold.factor == 2.5
    assert repr(copy) == repr(original)


def test_clone_produces_an_unfitted_copy(stepped: pd.Series) -> None:
    original = DeviationScorer().fit(stepped)
    copy = original.clone()
    assert original.fitted
    assert not copy.fitted
    assert copy is not original


def test_clone_does_not_share_nested_components() -> None:
    original = Pairing(DeviationScorer(), OverMean())
    copy = original.clone()
    assert copy.scorer is not original.scorer
    assert copy.threshold is not original.threshold


def test_repr_omits_defaults() -> None:
    assert repr(Doubler()) == "Doubler()"
    assert repr(Doubler(factor=3.0)) == "Doubler(factor=3.0)"


def test_repr_shows_required_parameters() -> None:
    rendered = repr(Pairing(DeviationScorer(), OverMean()))
    assert rendered.startswith("Pairing(scorer=DeviationScorer()")


def test_a_star_args_constructor_is_rejected() -> None:
    """Positional varargs cannot be named, so they cannot be round-tripped."""

    class Sloppy(Configurable):
        def __init__(self, *args: Any) -> None:
            self.args = args

    with pytest.raises(TypeError, match=r"takes \*args"):
        Sloppy(1, 2).get_params()


def test_keyword_varargs_are_ignored_rather_than_rejected() -> None:
    class Flexible(Configurable):
        def __init__(self, size: int = 1, **extra: Any) -> None:
            self.size = size
            self.extra = extra

    assert Flexible(size=4, colour="red").get_params() == {"size": 4}
