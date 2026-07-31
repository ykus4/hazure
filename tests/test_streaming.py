"""Tests for driving a fitted component one observation at a time.

The property worth most of this file is agreement: a stream that has seen the same
past as a batch run must produce the same numbers, because it is the same code
over a shorter window. That is asserted for several components, for a pipeline,
and — in the other direction — for the components that read forward and therefore
cannot agree with themselves online. The rest is about the buffer, the errors and
the round trip.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hazure import FixedThreshold, Pipeline, Stream, TimeSeries
from hazure.detection import (
    AutoregressionDetector,
    IqrDetector,
    LevelShiftDetector,
    SeasonalDetector,
    SpikeDetector,
)
from hazure.features import RollingAggregate
from hazure.methods import HampelDetector
from hazure.scoring import DeviationScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from collections.abc import Callable

# -- helpers ----------------------------------------------------------------


def noisy(periods: int = 240, *, name: str = "rps") -> pd.Series:
    """A regular hourly series around 100 with a daily cycle and one spike."""
    index = pd.date_range("2024-01-01", periods=periods, freq="h", name="time")
    rng = np.random.default_rng(0)
    cycle = 3.0 * np.sin(2 * np.pi * np.arange(periods) / 24)
    values = 100.0 + cycle + rng.normal(0, 0.5, periods)
    values[periods - 15] = 180.0
    return pd.Series(values, index=index, name=name)


def two_columns(periods: int = 200) -> pd.DataFrame:
    """Two independent hourly columns, so a univariate component fans out."""
    index = pd.date_range("2024-01-01", periods=periods, freq="h", name="time")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "a": 100.0 + rng.normal(0, 1, periods),
            "b": 50.0 + rng.normal(0, 1, periods),
        },
        index=index,
    )


def after(series: pd.Series | pd.DataFrame, hours: int = 1) -> pd.Timestamp:
    """The timestamp ``hours`` after the last one in ``series``."""
    return pd.Timestamp(series.index[-1]) + pd.Timedelta(hours=hours)


def spiky() -> tuple[pd.Series, SpikeDetector]:
    """A series and a spike detector already fitted on it."""
    series = noisy(200)
    return series, SpikeDetector(window=12).fit(series)


def _causal() -> list[tuple[str, Any, int]]:
    """Components whose answer for a row uses only that row and earlier ones.

    Built in a function rather than at import time so that a missing optional
    extra cannot stop this module from being collected. Each carries a history
    long enough for it, which is not simply its window: a pipeline consumes what
    every step consumes.
    """
    return [
        ("spike", SpikeDetector(window=24), 48),
        ("autoregression", AutoregressionDetector(n_steps=3), 24),
        ("hampel trailing", HampelDetector(window=11, center=False), 24),
        ("seasonal", SeasonalDetector(period=24), 72),
        ("iqr", IqrDetector(), 4),
        (
            "pipeline",
            Pipeline(
                [
                    ("rolling", RollingAggregate(window=6, agg="median")),
                    ("deviation", DeviationScorer()),
                    ("cut", IqrThreshold(factor=3.0)),
                ]
            ),
            24,
        ),
    ]


def _forward_looking() -> list[tuple[str, Any, int]]:
    """Components whose answer for a row reads rows after it."""
    return [
        # The window is centred by default, so half of it lies in the future.
        ("hampel centred", HampelDetector(window=11), 24),
        # Two windows, one either side of each point: the right one is forward.
        ("level shift", LevelShiftDetector(window=6), 24),
    ]


# -- the online answer is the batch answer ----------------------------------


@pytest.mark.parametrize(("name", "component", "history"), _causal())
def test_the_streaming_answer_equals_the_batch_answer(
    name: str, component: Any, history: int
) -> None:
    """The whole point: no second implementation, so no second answer."""
    whole = noisy(240)
    tail = 40
    fitted = component.fit(whole.iloc[:150])

    expected = fitted.run(TimeSeries.from_any(whole)).values[-tail:]
    stream = Stream(fitted, history=history).prime(whole.iloc[:-tail])
    streamed = stream.update_many(whole.iloc[-tail:])

    np.testing.assert_array_equal(
        np.asarray(streamed, dtype=float).ravel(), expected.ravel(), err_msg=name
    )


def test_one_observation_at_a_time_matches_a_streamed_batch() -> None:
    """``update_many`` is a loop over ``update``, and says so."""
    whole = noisy(200)
    fitted = SpikeDetector(window=12).fit(whole.iloc[:120])
    past, future = whole.iloc[:-10], whole.iloc[-10:]

    stepwise = Stream(fitted, history=24).prime(past)
    one_at_a_time = [
        stepwise.update(stamp, float(value)) for stamp, value in future.items()
    ]
    batched = Stream(fitted, history=24).prime(past).update_many(future)

    np.testing.assert_array_equal(
        np.asarray(one_at_a_time, dtype=float), np.asarray(batched, dtype=float)
    )


@pytest.mark.parametrize(("name", "component", "history"), _forward_looking())
def test_a_forward_looking_component_cannot_agree_with_its_batch_answer(
    name: str, component: Any, history: int
) -> None:
    """Nothing after the newest observation exists, so a centred window is NaN."""
    whole = noisy(200)
    tail = 20
    fitted = component.fit(whole.iloc[:120])

    expected = fitted.run(TimeSeries.from_any(whole)).values[-tail:]
    stream = Stream(fitted, history=history).prime(whole.iloc[:-tail], check=False)
    streamed = np.asarray(stream.update_many(whole.iloc[-tail:]), dtype=float)

    assert np.isnan(streamed).all(), name
    assert not np.isnan(expected).all(), name


def test_prime_refuses_a_forward_looking_component() -> None:
    """A centred window needs the future, and no buffer length supplies it.

    The last row of the primed history has no future in either the batch pass or
    the buffer, so probing only that row would find them in agreement. The check
    probes interior rows too, where the batch pass can see what came after.
    """
    series = noisy(120)
    with pytest.raises(ValueError, match="cannot be streamed: it reads"):
        Stream(HampelDetector(window=11), history=24).prime(series)


def test_a_forward_looking_component_can_still_be_streamed_without_the_check() -> None:
    """``check=False`` is the caller taking responsibility, and it is honoured."""
    series = noisy(120)
    stream = Stream(HampelDetector(window=11), history=24).prime(series, check=False)
    assert np.isnan(stream.update(after(series), 100.0))


# -- construction -----------------------------------------------------------


def test_streaming_needs_a_component() -> None:
    with pytest.raises(TypeError, match="needs a hazure Component"):
        Stream(object(), history=48)  # type: ignore[arg-type]


def test_history_must_hold_more_than_one_observation() -> None:
    _, detector = spiky()
    with pytest.raises(ValueError, match="keeps too little to compute anything"):
        Stream(detector, history=1)


def test_history_may_not_be_a_bool() -> None:
    """``True`` is an ``int``, and a buffer of one row is not what was meant."""
    _, detector = spiky()
    with pytest.raises(TypeError, match="not a bool"):
        Stream(detector, history=True)


def test_history_must_be_a_count_or_a_duration() -> None:
    _, detector = spiky()
    with pytest.raises(TypeError, match="neither a count of samples nor a duration"):
        Stream(detector, history=[24])  # type: ignore[arg-type]


def test_an_unreadable_duration_is_reported_by_the_duration_parser() -> None:
    """A string is the right type, so the complaint is about its value."""
    _, detector = spiky()
    with pytest.raises(ValueError, match="Unknown duration unit 'banana'"):
        Stream(detector, history="banana")


def test_a_duration_of_zero_is_refused() -> None:
    _, detector = spiky()
    with pytest.raises(ValueError, match="must be positive"):
        Stream(detector, history="0h")


def test_a_numpy_integer_counts_as_a_count_of_samples() -> None:
    _, detector = spiky()
    stream = Stream(detector, history=np.int64(24))
    assert stream.history == 24
    assert isinstance(stream.history, int)


# -- the fit has to have happened -------------------------------------------


def test_streaming_an_unfitted_component_is_refused() -> None:
    stream = Stream(SpikeDetector(window=6), history=12)
    with pytest.raises(RuntimeError, match="must be fitted"):
        stream.update("2024-01-01T00:00", {"rps": 100.0})


def test_priming_an_unfitted_component_is_refused() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        Stream(SpikeDetector(window=6), history=12).prime(noisy(100))


# -- prime ------------------------------------------------------------------


def test_prime_refuses_a_history_too_short_for_the_component() -> None:
    series = noisy(200)
    detector = SpikeDetector(window=24).fit(series)

    with pytest.raises(ValueError, match="too short for SpikeDetector") as raised:
        Stream(detector, history=5).prime(series)

    message = str(raised.value)
    # The message has to carry enough to act on: what was kept, out of what, and
    # the two answers that disagree.
    assert "retains 5 of the 200 observations" in message
    assert "comes out as nan rather than the 0" in message


def test_prime_returns_self_for_chaining() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=48)
    assert stream.prime(series) is stream


def test_prime_can_skip_the_check() -> None:
    """A history the check would reject is accepted when nobody asks."""
    series = noisy(200)
    detector = SpikeDetector(window=24).fit(series)
    stream = Stream(detector, history=5).prime(series, check=False)
    assert stream.buffer.n_rows == 5


def test_prime_skips_the_check_when_nothing_was_truncated() -> None:
    """Nothing was dropped, so there is no shorter answer to disagree with."""
    series = noisy(30)
    detector = SpikeDetector(window=24).fit(series)
    stream = Stream(detector, history=1000).prime(series)
    assert stream.buffer.n_rows == 30


# -- history as a duration --------------------------------------------------


def test_a_duration_history_retains_by_time_span() -> None:
    """A 12-hour span over hourly samples is 13 of them, ends included."""
    series = noisy(100)
    stream = Stream(IqrDetector().fit(series), history="12h").prime(series)
    assert stream.buffer.n_rows == 13


def test_a_duration_history_retains_by_time_span_on_an_irregular_axis() -> None:
    """A count of samples is not a length of time when the axis has gaps."""
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00",
            "2024-01-01 05:00",
            "2024-01-01 09:00",
            "2024-01-01 20:00",
            "2024-01-01 21:00",
            "2024-01-02 03:00",
        ],
        name="time",
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index, name="rps")

    stream = Stream(IqrDetector().fit(series), history="12h").prime(series)

    assert stream.buffer.n_rows == 3
    np.testing.assert_array_equal(stream.buffer.values.ravel(), [4.0, 5.0, 6.0])


def test_a_duration_history_survives_being_primed_with_nothing() -> None:
    """There is no newest observation to measure a span back from."""
    series = noisy(100)
    stream = Stream(IqrDetector().fit(series), history="6h").prime(series.iloc[:0])
    assert stream.buffer.n_rows == 0
    assert stream.n_seen == 0


def test_a_duration_history_trims_the_buffer_as_observations_arrive() -> None:
    series = noisy(100)
    stream = Stream(IqrDetector().fit(series), history="6h").prime(series)
    assert stream.buffer.n_rows == 7

    stream.update(after(series), 100.0)

    assert stream.buffer.n_rows == 7
    # The newest row is the one just pushed, and the oldest has fallen out.
    assert stream.buffer.time[-1] == after(series).value
    assert stream.buffer.time[0] == after(series, hours=-5).value


# -- time order -------------------------------------------------------------


def test_update_refuses_a_repeated_timestamp() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)
    with pytest.raises(ValueError, match="has to be fed in time order"):
        stream.update(series.index[-1], 100.0)


def test_update_refuses_an_earlier_timestamp() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)
    with pytest.raises(ValueError, match="went backwards"):
        stream.update(series.index[0], 100.0)


# -- reading one observation ------------------------------------------------


def test_update_reads_a_number_a_numpy_scalar_a_mapping_and_a_sequence() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)

    shapes: list[Any] = [100.0, np.float64(100.0), {"rps": 100.0}, [100.0]]
    verdicts = [
        stream.update(after(series, hours=i + 1), shape)
        for i, shape in enumerate(shapes)
    ]

    # The same number four ways, so the same verdict four times.
    assert verdicts == [0.0, 0.0, 0.0, 0.0]


def test_update_refuses_a_mapping_missing_a_tracked_column() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)
    with pytest.raises(ValueError, match=r"missing \['rps'\]"):
        stream.update(after(series), {"latency": 100.0})


def test_update_refuses_a_sequence_of_the_wrong_length() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)
    with pytest.raises(ValueError, match="given 2 values, but this component tracks 1"):
        stream.update(after(series), [100.0, 200.0])


def test_update_refuses_something_that_is_not_an_observation() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24).prime(series)
    with pytest.raises(TypeError, match="cannot read 'banana' as an observation"):
        stream.update(after(series), "banana")  # type: ignore[arg-type]


def test_update_refuses_one_number_for_a_multi_column_component() -> None:
    twin = two_columns()
    detector = SpikeDetector(window=12).fit(twin)
    stream = Stream(detector, history=24).prime(twin)
    with pytest.raises(ValueError, match=r"one number, but this component tracks"):
        stream.update(after(twin), 100.0)


def test_an_unprimed_stream_takes_a_bare_number_for_a_named_column() -> None:
    """One number means the one column the component was fitted on, whatever its name.

    Nothing has told the stream the column name yet, so it takes the fitted one
    rather than inventing a placeholder that would then fail to match it.
    """
    series, detector = spiky()
    stream = Stream(detector, history=24)
    assert np.isnan(stream.update(after(series), 100.0))
    assert stream.columns == ("rps",)


# -- several columns --------------------------------------------------------


def test_a_stream_fans_out_over_the_columns_a_component_was_fitted_on() -> None:
    twin = two_columns()
    detector = SpikeDetector(window=12).fit(twin)
    assert detector.feature_names == ("a", "b")

    stream = Stream(detector, history=24).prime(twin)
    verdict = stream.update(after(twin), {"a": 400.0, "b": 50.0})

    assert isinstance(verdict, dict)
    assert set(verdict) == {"a", "b"}
    assert verdict == {"a": 1.0, "b": 0.0}


def test_update_many_returns_one_verdict_column_per_tracked_column() -> None:
    twin = two_columns()
    detector = SpikeDetector(window=12).fit(twin)
    stream = Stream(detector, history=24).prime(twin)

    later = pd.DataFrame(
        {"a": [400.0, 100.0], "b": [50.0, 50.0]},
        index=pd.date_range(after(twin), periods=2, freq="h", name="time"),
    )
    verdicts = stream.update_many(later)

    assert list(verdicts.columns) == ["a", "b"]
    np.testing.assert_array_equal(verdicts["a"].to_numpy(), [1.0, 0.0])
    np.testing.assert_array_equal(verdicts["b"].to_numpy(), [0.0, 0.0])


def test_a_too_short_history_reports_every_column_that_disagrees() -> None:
    twin = two_columns()
    detector = SpikeDetector(window=24).fit(twin)
    with pytest.raises(ValueError, match=r"comes out as \[nan, nan\] rather than"):
        Stream(detector, history=5).prime(twin)


def test_a_stream_refuses_columns_the_component_was_not_fitted_on() -> None:
    series, detector = spiky()
    other = pd.DataFrame({"latency": np.arange(30.0)}, index=series.index[:30])
    with pytest.raises(ValueError, match=r"\['rps'\] is not among the \['latency'\]"):
        Stream(detector, history=24).prime(other)


def test_update_many_refuses_columns_the_component_was_not_fitted_on() -> None:
    series, detector = spiky()
    other = pd.DataFrame({"latency": np.arange(30.0)}, index=series.index[:30])
    with pytest.raises(ValueError, match=r"was fitted on \['rps'\]"):
        Stream(detector, history=24).update_many(other)


# -- state ------------------------------------------------------------------


def test_update_answers_nan_until_the_buffer_has_filled() -> None:
    """Without priming there is nothing to judge against, and it says so."""
    series, detector = spiky()
    stream = Stream(detector, history=24)

    stamps = pd.date_range(after(series), periods=3, freq="h")
    verdicts = [stream.update(stamp, {"rps": 100.0}) for stamp in stamps]

    assert all(np.isnan(verdict) for verdict in verdicts)


def test_columns_is_none_before_the_first_observation() -> None:
    _, detector = spiky()
    assert Stream(detector, history=24).columns is None


def test_the_first_observation_settles_the_columns() -> None:
    series, detector = spiky()
    stream = Stream(detector, history=24)
    stream.update(after(series), {"rps": 100.0})
    assert stream.columns == ("rps",)


def test_n_seen_counts_primed_rows_and_updates() -> None:
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    stream = Stream(detector, history=12)
    assert stream.n_seen == 0

    stream.prime(series)
    assert stream.n_seen == 100

    stream.update(after(series), 100.0)
    assert stream.n_seen == 101


def test_buffer_holds_the_retained_rows() -> None:
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    stream = Stream(detector, history=12).prime(series)

    buffered = stream.buffer

    assert isinstance(buffered, TimeSeries)
    assert buffered.n_rows == 12
    assert buffered.columns == ("rps",)
    np.testing.assert_array_equal(
        buffered.values.ravel(), series.to_numpy(dtype=float)[-12:]
    )


def test_repr_mentions_the_buffered_count() -> None:
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    rendered = repr(Stream(detector, history=12).prime(series))
    assert "buffered=12" in rendered
    assert "n_seen=100" in rendered


# -- batches ----------------------------------------------------------------


def test_update_many_returns_the_backend_it_was_given(
    backend: str, native_factory: Callable[..., Any]
) -> None:
    rng = np.random.default_rng(2)
    past = native_factory(100.0 + rng.normal(0, 1, 60))
    detector = SpikeDetector(window=6).fit(past)
    stream = Stream(detector, history=12).prime(past)

    later = native_factory(100.0 + rng.normal(0, 1, 5), start="2024-01-04")
    verdicts = stream.update_many(later)

    assert type(verdicts) is type(later), backend
    assert len(verdicts) == 5


def test_update_many_accepts_an_empty_batch() -> None:
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    stream = Stream(detector, history=12).prime(series)

    verdicts = stream.update_many(series.iloc[:0])

    assert len(verdicts) == 0
    assert stream.n_seen == 100


# -- serialisation ----------------------------------------------------------


def test_a_primed_stream_survives_a_round_trip_through_json() -> None:
    """A running monitor is stored whole: the fit, and the past it was judging."""
    series = noisy(200)
    detector = SpikeDetector(window=12).fit(series)
    stream = Stream(detector, history=24).prime(series)

    resumed = Stream.from_dict(json.loads(json.dumps(stream.to_dict())))

    assert isinstance(resumed.component, SpikeDetector)
    assert resumed.component.fitted
    assert resumed.history == 24
    assert resumed.n_seen == stream.n_seen
    assert resumed.columns == stream.columns
    np.testing.assert_array_equal(resumed.buffer.time, stream.buffer.time)
    np.testing.assert_array_equal(resumed.buffer.values, stream.buffer.values)

    # The state came back, not merely the parameters: the resumed stream places
    # the next observation exactly as the original would have.
    stamp = after(series)
    assert resumed.update(stamp, 400.0) == stream.update(stamp, 400.0) == 1.0


def test_a_timedelta_history_cannot_be_serialised() -> None:
    """A documented limitation: a timedelta has no JSON representation."""
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    stream = Stream(detector, history=timedelta(hours=12)).prime(series, check=False)
    with pytest.raises(TypeError, match="cannot serialise"):
        stream.to_dict()


def test_a_string_history_serialises_where_a_timedelta_does_not() -> None:
    series = noisy(100)
    detector = SpikeDetector(window=6).fit(series)
    stream = Stream(detector, history="12h").prime(series)
    resumed = Stream.from_dict(json.loads(json.dumps(stream.to_dict())))
    assert resumed.history == "12h"
    assert resumed.buffer.n_rows == stream.buffer.n_rows


def test_a_bare_number_falls_back_to_a_placeholder_name() -> None:
    """With nothing fitted there is no name to prefer, so one is invented."""
    stream = Stream(FixedThreshold(high=1.0), history=4)
    assert stream.update("2024-01-01T00:00", 0.5) == 0.0
    assert stream.columns == ("value",)


def test_a_bare_sequence_falls_back_to_placeholder_names() -> None:
    """Same for several values, which need telling apart as well as naming."""
    stream = Stream(FixedThreshold(high=1.0), history=4)
    assert stream.update("2024-01-01T00:00", [0.5, 9.0]) == {
        "value_0": 0.0,
        "value_1": 1.0,
    }
    assert stream.columns == ("value_0", "value_1")
