"""Tests for the aggregators.

The three-valued logic is the substance here, so every cell of every truth table
is asserted rather than sampled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hazure import TimeSeries
from hazure.ensemble import (
    AndAggregator,
    CustomizedAggregator,
    OrAggregator,
    VoteAggregator,
)
from tests.conftest import BACKENDS, make_native

if TYPE_CHECKING:
    from hazure import BaseAggregator

NAN = np.nan

#: Every combination of two labels, in a fixed order the expectations mirror.
PAIRS = [
    (1.0, 1.0),
    (1.0, 0.0),
    (1.0, NAN),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.0, NAN),
    (NAN, 1.0),
    (NAN, 0.0),
    (NAN, NAN),
]


def combine(aggregator: BaseAggregator, columns: Any) -> np.ndarray:
    """Aggregate a matrix whose columns are label series, returning the labels."""
    values = np.asarray(columns, dtype=float)
    time = np.datetime64("2024-01-01") + np.arange(values.shape[0]) * np.timedelta64(
        1, "h"
    )
    ts = TimeSeries.from_arrays(
        time, values, [f"input_{i}" for i in range(values.shape[1])]
    )
    result = aggregator.aggregate(ts)
    return np.asarray(result["anomaly"], dtype=float)


# -- truth tables -----------------------------------------------------------


def test_or_is_anomalous_when_any_input_says_so() -> None:
    expected = [1.0, 1.0, 1.0, 1.0, 0.0, NAN, 1.0, NAN, NAN]
    np.testing.assert_array_equal(combine(OrAggregator(), PAIRS), expected)


def test_and_is_anomalous_only_when_every_input_agrees() -> None:
    expected = [1.0, 0.0, NAN, 0.0, 0.0, 0.0, NAN, 0.0, NAN]
    np.testing.assert_array_equal(combine(AndAggregator(), PAIRS), expected)


def test_a_majority_vote_ignores_unknown_inputs() -> None:
    """Unknown labels abstain, so a tie among the known ones still carries."""
    expected = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, NAN]
    np.testing.assert_array_equal(combine(VoteAggregator(), PAIRS), expected)


def test_a_unanimous_vote_needs_every_known_input() -> None:
    expected = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, NAN]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=1.0), PAIRS), expected
    )


def test_a_vote_of_zero_flags_every_row_with_a_known_label() -> None:
    expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, NAN]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=0.0), PAIRS), expected
    )


def test_a_vote_counts_the_exact_fraction_of_three_inputs() -> None:
    """One of three is 1/3, two of three is 2/3, either side of a 0.5 threshold."""
    rows = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, NAN),
    ]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=0.5), rows), [0.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=2 / 3), rows), [0.0, 1.0, 1.0, 0.0]
    )


def test_an_all_unknown_row_is_unknown_for_every_aggregator() -> None:
    for aggregator in (OrAggregator(), AndAggregator(), VoteAggregator()):
        assert np.isnan(combine(aggregator, [(NAN, NAN, NAN)])[0])


def test_a_non_binary_label_counts_as_anomalous() -> None:
    """A detector emitting counts or confidences still aggregates sensibly."""
    np.testing.assert_array_equal(
        combine(AndAggregator(), [(3.0, 0.5), (3.0, 0.0)]), [1.0, 0.0]
    )


# -- shape and naming -------------------------------------------------------


def test_every_aggregator_emits_a_single_column_named_anomaly() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=index)
    for aggregator in (OrAggregator(), AndAggregator(), VoteAggregator()):
        result = aggregator.aggregate(frame)
        assert list(result.columns) == ["anomaly"]


def test_aggregating_separate_series_joins_them_first() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    left = pd.Series([1.0, 1.0, 0.0, 0.0], index=index, name="flag")
    right = pd.Series([1.0, 0.0, 1.0, 0.0], index=index, name="flag")
    combined = AndAggregator().aggregate(left, right)
    assert list(combined) == [1.0, 0.0, 0.0, 0.0]
    assert combined.name == "anomaly"


def test_a_single_row_is_aggregated_like_any_other() -> None:
    np.testing.assert_array_equal(combine(OrAggregator(), [(0.0, 1.0)]), [1.0])


# -- customized aggregation -------------------------------------------------


def test_a_customized_aggregator_receives_the_label_matrix() -> None:
    seen: list[tuple[int, ...]] = []

    def first_input(labels: np.ndarray) -> np.ndarray:
        seen.append(labels.shape)
        return labels[:, 0]

    result = combine(CustomizedAggregator(aggregate_func=first_input), PAIRS)
    assert seen == [(9, 2)]
    np.testing.assert_array_equal(result, [row[0] for row in PAIRS])


def test_a_customized_aggregator_passes_its_parameters() -> None:
    weighted = CustomizedAggregator(
        aggregate_func=lambda labels, w: (labels @ w >= 0.5).astype(float),
        aggregate_func_params={"w": np.array([0.67, 0.33])},
    )
    np.testing.assert_array_equal(
        combine(weighted, [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]), [1.0, 0.0, 0.0]
    )


def test_a_customized_aggregator_accepts_a_single_column_result() -> None:
    reshaped = CustomizedAggregator(aggregate_func=lambda labels: labels[:, :1].copy())
    np.testing.assert_array_equal(
        combine(reshaped, [(1.0, 0.0), (0.0, 0.0)]), [1.0, 0.0]
    )


def test_a_customized_aggregator_rejects_a_wrong_row_count() -> None:
    shrinking = CustomizedAggregator(aggregate_func=lambda labels: labels[:1, 0])
    with pytest.raises(ValueError, match=r"must give shape \(2,\)"):
        combine(shrinking, [(1.0, 0.0), (0.0, 0.0)])


# -- parameters -------------------------------------------------------------


def test_the_quantifying_aggregators_have_no_parameters() -> None:
    assert OrAggregator().get_params() == {}
    assert AndAggregator().get_params() == {}
    assert repr(AndAggregator()) == "AndAggregator()"


def test_a_vote_exposes_its_threshold() -> None:
    assert VoteAggregator(threshold=0.75).get_params() == {"threshold": 0.75}
    assert repr(VoteAggregator(threshold=0.75)) == "VoteAggregator(threshold=0.75)"


def test_clone_round_trips_every_aggregator_parameter() -> None:
    def half(labels: np.ndarray) -> np.ndarray:
        return labels[:, 0]

    originals: list[BaseAggregator] = [
        OrAggregator(),
        AndAggregator(),
        VoteAggregator(threshold=0.8),
        CustomizedAggregator(aggregate_func=half, aggregate_func_params={"unused": 1}),
    ]
    for original in originals:
        copy = original.clone()
        assert copy.get_params() == original.get_params()
        assert repr(copy) == repr(original)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_a_vote_threshold_must_be_a_fraction(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"threshold must be a fraction"):
        combine(VoteAggregator(threshold=threshold), PAIRS)


# -- backend independence ---------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_aggregator_returns_the_backend_it_was_given(backend: str) -> None:
    left = make_native(backend, [1.0, 0.0, 1.0], name="left")
    right = make_native(backend, [1.0, 1.0, 0.0], name="right")
    result = OrAggregator().aggregate(left, right)
    assert type(result) is type(left)


def test_every_backend_produces_the_same_labels() -> None:
    left_values = [1.0, 0.0, np.nan, 1.0]
    right_values = [1.0, 1.0, 0.0, np.nan]
    results = [
        TimeSeries.from_any(
            VoteAggregator(threshold=0.5).aggregate(
                make_native(backend, left_values, name="left"),
                make_native(backend, right_values, name="right"),
            )
        )
        for backend in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)
    np.testing.assert_array_equal(results[0].values.ravel(), [1.0, 1.0, 0.0, 1.0])
