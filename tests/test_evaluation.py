"""Tests for hazure.evaluation: point-based and event-based scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hazure.evaluation import f1_score, iou, precision, recall
from hazure.events import Events, to_events
from tests.conftest import BACKENDS, make_native

HOUR = 3_600_000_000_000


def labels(flags: list[float], start: str = "2024-01-01") -> pd.Series:
    """An hourly pandas label series."""
    index = pd.date_range(start, periods=len(flags), freq="h", name="time")
    return pd.Series(flags, index=index, name="x")


# ---------------------------------------------------------------------------
# point-based
# ---------------------------------------------------------------------------


def test_point_recall_counts_true_points_that_were_found() -> None:
    # 3 true points, 2 of them flagged.
    assert recall(labels([1, 1, 1, 0, 0]), labels([1, 1, 0, 1, 0])) == pytest.approx(
        2 / 3
    )


def test_point_precision_counts_flags_that_were_right() -> None:
    # 3 flags, 2 of them on true points.
    assert precision(labels([1, 1, 1, 0, 0]), labels([1, 1, 0, 1, 0])) == pytest.approx(
        2 / 3
    )


def test_point_f1_is_the_harmonic_mean() -> None:
    truth, guess = labels([1, 1, 1, 0, 0]), labels([1, 0, 0, 1, 0])
    # recall 1/3, precision 1/2 -> 2 * (1/6) / (5/6) = 0.4
    assert f1_score(truth, guess) == pytest.approx(0.4)


def test_point_iou_divides_shared_points_by_flagged_points() -> None:
    # Both flag 2 points; either flags 4.
    assert iou(labels([1, 1, 1, 0, 0]), labels([1, 1, 0, 1, 0])) == 0.5


def test_a_perfect_prediction_scores_one_everywhere() -> None:
    truth = labels([0, 1, 1, 0, 1])
    assert recall(truth, truth) == 1.0
    assert precision(truth, truth) == 1.0
    assert f1_score(truth, truth) == 1.0
    assert iou(truth, truth) == 1.0


def test_point_metrics_read_nan_labels_as_not_anomalous() -> None:
    truth = labels([1, 1, np.nan, 0, 0])
    guess = labels([1, np.nan, 1, 0, 0])
    # truth marks 2 points, guess marks 2, they share 1.
    assert recall(truth, guess) == 0.5
    assert precision(truth, guess) == 0.5
    assert iou(truth, guess) == pytest.approx(1 / 3)


def test_point_recall_is_undefined_when_nothing_is_anomalous() -> None:
    assert math.isnan(float(recall(labels([0, 0, 0]), labels([1, 0, 0]))))


def test_point_precision_is_undefined_when_nothing_was_flagged() -> None:
    assert math.isnan(float(precision(labels([1, 0, 0]), labels([0, 0, 0]))))


def test_point_iou_is_undefined_when_neither_marks_anything() -> None:
    assert math.isnan(float(iou(labels([0, 0, 0]), labels([0, 0, 0]))))


def test_f1_is_undefined_when_precision_and_recall_are_both_zero() -> None:
    assert math.isnan(float(f1_score(labels([1, 0, 0]), labels([0, 1, 0]))))


def test_f1_is_undefined_when_recall_is_undefined() -> None:
    assert math.isnan(float(f1_score(labels([0, 0, 0]), labels([1, 0, 0]))))


def test_point_metrics_reject_mismatched_time_axes() -> None:
    truth = labels([1, 1, 0])
    guess = labels([1, 1, 0], start="2024-01-02")
    with pytest.raises(ValueError, match="must share a time axis"):
        recall(truth, guess)
    with pytest.raises(ValueError, match="must share a time axis"):
        iou(truth, guess)


def test_point_metrics_accept_an_identical_axis_in_a_different_flavour() -> None:
    truth = make_native("pandas", [1.0, 1.0, 0.0, 0.0])
    guess = make_native("polars", [1.0, 0.0, 0.0, 0.0])
    assert recall(truth, guess) == 0.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_metrics_give_the_same_number_on_every_backend(backend: str) -> None:
    truth = make_native(backend, [1.0, 1.0, 1.0, 0.0, 0.0])
    guess = make_native(backend, [1.0, 1.0, 0.0, 1.0, 0.0])
    assert recall(truth, guess) == pytest.approx(2 / 3)
    assert precision(truth, guess) == pytest.approx(2 / 3)
    assert iou(truth, guess) == 0.5


# ---------------------------------------------------------------------------
# point-based, multi-column
# ---------------------------------------------------------------------------


def two_column(
    left: list[float], right: list[float], names: tuple[str, str] = ("a", "b")
) -> pd.DataFrame:
    """A two-column hourly label frame."""
    index = pd.date_range("2024-01-01", periods=len(left), freq="h", name="time")
    return pd.DataFrame({names[0]: left, names[1]: right}, index=index)


def test_a_multi_column_input_is_scored_one_column_at_a_time() -> None:
    truth = two_column([1, 1, 0, 0], [1, 0, 0, 0])
    guess = two_column([1, 0, 0, 0], [1, 1, 0, 0])
    assert recall(truth, guess) == {"a": 0.5, "b": 1.0}
    assert precision(truth, guess) == {"a": 1.0, "b": 0.5}
    assert iou(truth, guess) == {"a": 0.5, "b": 0.5}


def test_a_multi_column_f1_uses_each_column_s_own_precision_and_recall() -> None:
    truth = two_column([1, 1, 0, 0], [1, 0, 0, 0])
    guess = two_column([1, 0, 0, 0], [1, 1, 0, 0])
    scores = f1_score(truth, guess)
    assert isinstance(scores, dict)
    assert scores["a"] == pytest.approx(2 / 3)
    assert scores["b"] == pytest.approx(2 / 3)


def test_multi_column_metrics_reject_differing_column_names() -> None:
    truth = two_column([1, 0], [1, 0], names=("a", "b"))
    guess = two_column([1, 0], [1, 0], names=("a", "c"))
    with pytest.raises(ValueError, match=r"same columns.*\['b', 'c'\]"):
        recall(truth, guess)


# ---------------------------------------------------------------------------
# event-based
# ---------------------------------------------------------------------------


def test_an_event_is_detected_once_enough_of_it_is_covered() -> None:
    # A ten-hour outage with four of its hours flagged: 0.4 of its duration.
    index = pd.date_range("2024-01-01", periods=11, freq="h")
    truth = to_events(pd.Series([1.0] * 10 + [0.0], index=index))
    guess = to_events(pd.Series([1.0] * 4 + [0.0] * 7, index=index))
    assert recall(truth, guess) == 0.0
    assert recall(truth, guess, 0.4) == 1.0
    # The whole prediction sits inside the outage, so it is entirely justified.
    assert precision(truth, guess) == 1.0


def test_coverage_is_summed_across_every_overlapping_prediction() -> None:
    # 30ns + 40ns of a 100ns event is 0.7, over the default threshold.
    truth = Events.from_bounds([[0, 99]])
    guess = Events.from_bounds([[0, 29], [60, 99]])
    assert recall(truth, guess) == 1.0
    assert recall(truth, guess, 0.8) == 0.0
    # Both predictions are fully inside the true event.
    assert precision(truth, guess) == 1.0


@pytest.mark.parametrize(
    ("m_samples", "k_samples"),
    [(2, 1), (4, 1), (4, 3), (8, 3), (8, 5), (5, 2), (10, 7), (3, 1)],
)
def test_covering_k_of_m_samples_is_a_coverage_of_exactly_k_over_m(
    m_samples: int, k_samples: int
) -> None:
    # The property the corrected duration arithmetic exists for: a true event of
    # m samples with exactly k of them predicted is a hit at a threshold of k/m
    # and a miss at anything above it, with no nanosecond of slack either way.
    index = pd.date_range("2024-01-01", periods=m_samples + 1, freq="h")
    truth = to_events(pd.Series([1.0] * m_samples + [0.0], index=index))
    guess = to_events(
        pd.Series([1.0] * k_samples + [0.0] * (m_samples + 1 - k_samples), index=index)
    )
    assert truth.total_duration == m_samples * HOUR
    assert truth.intersect(guess).total_duration == k_samples * HOUR

    exact = k_samples / m_samples
    assert recall(truth, guess, exact) == 1.0
    assert recall(truth, guess, exact + 1e-9) == 0.0


def test_event_recall_is_the_fraction_of_true_events_hit() -> None:
    truth = Events.from_bounds([[0, 100], [200, 300], [400, 500]])
    guess = Events.from_bounds([[0, 100]])
    assert recall(truth, guess) == pytest.approx(1 / 3)
    assert precision(truth, guess) == 1.0


def test_event_f1_takes_independent_thresholds() -> None:
    truth = Events.from_bounds([[0, 99]])
    guess = Events.from_bounds([[0, 29], [1000, 1099]])
    # recall: 30 of 100ns covered, so a 0.3 threshold catches it, 0.5 does not.
    # precision: [0, 29] is fully covered, [1000, 1099] not at all -> 0.5.
    assert recall(truth, guess, 0.3) == 1.0
    assert recall(truth, guess) == 0.0
    assert precision(truth, guess) == 0.5
    assert f1_score(truth, guess, recall_thresh=0.3) == pytest.approx(2 / 3)
    assert f1_score(truth, guess) == 0.0


def test_an_instantaneous_true_event_is_hit_by_any_prediction_covering_it() -> None:
    # No special case is needed: the instant lasts one nanosecond, so a
    # prediction containing it covers all of it and one that misses covers none.
    truth = Events.from_bounds([[50, 50]])
    assert recall(truth, Events.from_bounds([[0, 100]])) == 1.0
    assert recall(truth, Events.from_bounds([[50, 50]])) == 1.0
    assert recall(truth, Events.from_bounds([[0, 49]])) == 0.0
    assert recall(truth, Events.from_bounds([[51, 100]])) == 0.0
    assert recall(truth, Events.empty()) == 0.0


def test_instantaneous_and_lasting_true_events_are_judged_side_by_side() -> None:
    truth = Events.from_bounds([[0, 0], [100, 200]])
    guess = Events.from_bounds([[100, 200]])
    assert recall(truth, guess) == 0.5


def test_event_iou_divides_shared_duration_by_covered_duration() -> None:
    # 18 of 24 anomalous hours in common, 24 in either -- exactly, because a
    # period event spanning k samples lasts exactly k steps.
    index = pd.date_range("2024-01-01", periods=25, freq="h")
    truth = to_events(pd.Series([1.0] * 24 + [0.0], index=index))
    guess = to_events(pd.Series([1.0] * 18 + [0.0] * 7, index=index))
    assert iou(truth, guess) == 0.75


def test_event_iou_agrees_with_the_point_based_score_on_period_events() -> None:
    truth = labels([1, 1, 1, 1, 0])
    guess = labels([1, 1, 1, 0, 0])
    assert iou(to_events(truth), to_events(guess)) == iou(truth, guess) == 0.75


def test_event_iou_ignores_the_threshold_entirely() -> None:
    truth = Events.from_bounds([[0, 99]])
    guess = Events.from_bounds([[0, 9]])
    assert iou(truth, guess) == 0.1


def test_event_iou_of_two_disjoint_sets_is_zero() -> None:
    assert iou(Events.from_bounds([[0, 10]]), Events.from_bounds([[100, 110]])) == 0.0


def test_event_recall_is_undefined_when_there_are_no_true_events() -> None:
    assert math.isnan(float(recall([], [(0, 10)])))


def test_event_precision_is_undefined_when_nothing_was_predicted() -> None:
    assert math.isnan(float(precision([(0, 10)], [])))


def test_event_iou_is_undefined_only_when_there_are_no_events_at_all() -> None:
    assert math.isnan(float(iou([], [])))


def test_two_identical_instants_agree_completely() -> None:
    # Every event lasts at least a nanosecond, so this is a real 1.0 rather
    # than an undefined 0/0.
    assert iou([5], [5]) == 1.0
    assert recall([5], [5]) == 1.0
    assert precision([5], [5]) == 1.0


def test_event_metrics_accept_an_events_against_a_plain_list() -> None:
    assert recall(Events.from_bounds([[0, 100]]), [(0, 100)]) == 1.0


def test_event_metrics_merge_overlapping_input_intervals_first() -> None:
    # The two overlapping true intervals are one event, fully covered.
    assert recall([(0, 100), (50, 150)], [(0, 150)]) == 1.0


# ---------------------------------------------------------------------------
# dicts
# ---------------------------------------------------------------------------


def test_a_dict_is_scored_key_by_key() -> None:
    truth = {"spike": [(0, 100)], "shift": [(0, 100)]}
    guess = {"spike": [(0, 100)], "shift": [(200, 300)]}
    assert recall(truth, guess) == {"spike": 1.0, "shift": 0.0}
    assert iou(truth, guess) == {"spike": 1.0, "shift": 0.0}


def test_thresh_reaches_every_key_of_a_dict() -> None:
    # 30 of 100 covered in both keys: the default threshold misses, 0.3 hits.
    truth = {"a": [(0, 100)], "b": [(0, 100)]}
    guess = {"a": [(0, 30)], "b": [(0, 30)]}
    assert recall(truth, guess) == {"a": 0.0, "b": 0.0}
    assert recall(truth, guess, 0.3) == {"a": 1.0, "b": 1.0}
    assert precision(truth, guess, 0.3) == {"a": 1.0, "b": 1.0}


def test_both_f1_thresholds_reach_every_key_of_a_dict() -> None:
    truth = {"a": [(0, 100)], "b": [(0, 100)]}
    guess = {"a": [(0, 30), (1000, 1100)], "b": [(0, 30), (1000, 1100)]}
    default = f1_score(truth, guess)
    assert default == {"a": 0.0, "b": 0.0}
    tolerant = f1_score(truth, guess, recall_thresh=0.3)
    assert isinstance(tolerant, dict)
    assert tolerant["a"] == pytest.approx(2 / 3)
    assert tolerant["b"] == pytest.approx(2 / 3)


def test_a_dict_of_label_series_is_scored_key_by_key() -> None:
    truth = {"a": labels([1, 1, 0, 0]), "b": labels([1, 0, 0, 0])}
    guess = {"a": labels([1, 0, 0, 0]), "b": labels([1, 1, 0, 0])}
    assert recall(truth, guess) == {"a": 0.5, "b": 1.0}


def test_dict_metrics_reject_differing_keys() -> None:
    with pytest.raises(ValueError, match=r"same keys.*\['b', 'c'\]"):
        recall({"a": [(0, 1)], "b": [(0, 1)]}, {"a": [(0, 1)], "c": [(0, 1)]})


def test_dict_metrics_reject_a_key_holding_several_anomaly_types() -> None:
    truth = {"a": two_column([1, 0], [1, 0])}
    guess = {"a": two_column([1, 0], [1, 0])}
    with pytest.raises(TypeError, match="must hold a single anomaly type"):
        recall(truth, guess)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


def test_metrics_reject_mixing_events_with_labels() -> None:
    with pytest.raises(TypeError, match="y_true is events but y_pred is labels"):
        recall([(0, 10)], labels([1, 0, 0]))
    with pytest.raises(TypeError, match="y_true is labels but y_pred is mapping"):
        iou(labels([1, 0, 0]), {"a": [(0, 10)]})


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_metrics_reject_a_threshold_outside_the_half_open_unit_interval(
    bad: float,
) -> None:
    events = Events.from_bounds([[0, 10]])
    with pytest.raises(ValueError, match="thresh must be greater than 0"):
        recall(events, events, bad)
    with pytest.raises(ValueError, match="thresh must be greater than 0"):
        precision(events, events, bad)
    with pytest.raises(ValueError, match="recall_thresh must be greater than 0"):
        f1_score(events, events, recall_thresh=bad)
    with pytest.raises(ValueError, match="precision_thresh must be greater than 0"):
        f1_score(events, events, precision_thresh=bad)


def test_a_threshold_of_exactly_one_demands_complete_coverage() -> None:
    truth = Events.from_bounds([[0, 100]])
    assert recall(truth, Events.from_bounds([[0, 100]]), 1.0) == 1.0
    assert recall(truth, Events.from_bounds([[0, 99]]), 1.0) == 0.0


def test_metrics_reject_a_non_numeric_threshold() -> None:
    events = Events.from_bounds([[0, 10]])
    with pytest.raises(TypeError, match="thresh must be a number"):
        recall(events, events, "0.5")  # type: ignore[arg-type]
