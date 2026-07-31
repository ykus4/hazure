"""Tests for hazure.datasets: the generator, the NAB loader and the compare loop.

Two things shape this file. The first is that every planted anomaly is checked
against the detector built for its shape, and against one built for a different
shape that should therefore miss it — "a spike detector finds spikes" is only
interesting next to "a distribution rule does not find a level shift".

The second is that nothing here touches the network. The NAB loader is exercised
either against a hand-written cache under ``tmp_path`` or against a monkeypatched
``urlopen``; a test that did neither would pass on a laptop and hang in CI.
"""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from urllib.error import URLError

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from hazure import TimeSeries
from hazure.datasets import (
    KINDS,
    Dataset,
    compare,
    load_nab,
    make_series,
    nab_names,
)
from hazure.datasets import nab as nab_module
from hazure.detection import (
    IqrDetector,
    LevelShiftDetector,
    SeasonalDetector,
    SpikeDetector,
    ThresholdDetector,
    VolatilityShiftDetector,
)
from hazure.evaluation import recall
from hazure.events import to_events
from hazure.scoring import DeviationScorer

HOUR = 3_600_000_000_000
COMPARE_KEYS = {"recall", "precision", "f1", "alerts", "delay"}

# -- helpers ----------------------------------------------------------------


def values_of(dataset: Dataset) -> np.ndarray:
    """The dataset's values as a flat float array, whatever backend it is in."""
    return np.asarray(TimeSeries.from_any(dataset.data).values, dtype=float).ravel()


def labels_of(dataset: Dataset) -> np.ndarray:
    """The dataset's ground truth as a flat float array on its own axis."""
    return np.asarray(TimeSeries.from_any(dataset.labels).values, dtype=float).ravel()


def found_by(detector: Any, dataset: Dataset) -> Any:
    """Fit the detector on the dataset and convert its labels to events."""
    return to_events(detector.fit_detect(dataset.data))


def recall_of(detector: Any, dataset: Dataset, thresh: float = 0.5) -> float:
    """Event-based recall of one detector against one dataset's ground truth."""
    return float(recall(dataset.events, found_by(detector, dataset), thresh))


def kind_kwargs(kind: str) -> dict[str, Any]:
    """The extra arguments a kind insists on, so every kind can be built alike."""
    return {"period": 24} if kind == "seasonal_break" else {}


def nab_rows(n_rows: int, *, fractional: bool = False) -> str:
    """One NAB-shaped CSV: hourly ``timestamp,value`` with the values counting up."""
    tail = ".000000" if fractional else ""
    body = [
        f"2024-01-01 {hour:02d}:00:00{tail},{float(hour)}" for hour in range(n_rows)
    ]
    return "\n".join(["timestamp,value", *body]) + "\n"


def write_cache(
    root: Path,
    windows: dict[str, list[list[str]]],
    files: dict[str, str],
) -> None:
    """Lay out a fake NAB cache: the label file, and one CSV per named series."""
    labels = root / "labels" / "combined_windows.json"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps(windows))
    for key, text in files.items():
        target = root / "data" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


class FakeResponse:
    """The bit of ``urlopen``'s contract that :mod:`hazure.datasets.nab` uses."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class RememberFit:
    """A detector that flags nothing and remembers how much it was fitted on."""

    def __init__(self) -> None:
        self.fitted_rows: int | None = None

    def fit(self, data: Any) -> RememberFit:
        self.fitted_rows = TimeSeries.from_any(data).n_rows
        return self

    def detect(self, data: Any) -> Any:
        ts = TimeSeries.from_any(data)
        return ts.wrap(np.zeros(ts.n_rows))

    def fit_detect(self, data: Any) -> Any:
        return self.fit(data).detect(data)


# ---------------------------------------------------------------------------
# make_series: shape of the output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_plants_exactly_the_number_of_anomalies_asked_for(
    kind: str,
) -> None:
    dataset = make_series(kind, n=1500, n_anomalies=4, **kind_kwargs(kind))
    assert dataset.events.n_events == 4
    assert len(values_of(dataset)) == 1500
    assert dataset.name == kind


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_describes_itself_in_the_description(kind: str) -> None:
    dataset = make_series(kind, n=1000, n_anomalies=2, **kind_kwargs(kind))
    assert kind in dataset.description
    assert KINDS[kind] in dataset.description


def test_the_same_arguments_give_byte_identical_values() -> None:
    first = make_series("spike", n=500, n_anomalies=2)
    second = make_series("spike", n=500, n_anomalies=2)
    assert values_of(first).tobytes() == values_of(second).tobytes()


def test_a_different_seed_gives_different_values() -> None:
    first = make_series("spike", n=500, n_anomalies=2, seed=0)
    second = make_series("spike", n=500, n_anomalies=2, seed=1)
    assert values_of(first).tobytes() != values_of(second).tobytes()
    # Only the noise moved: the anomalies are placed by position, not by chance.
    assert first.events == second.events


def test_the_default_return_is_a_timeseries() -> None:
    assert isinstance(make_series("spike", n=200, n_anomalies=1).data, TimeSeries)


def test_a_backend_argument_emits_that_backend(backend: str) -> None:
    dataset = make_series("spike", n=200, n_anomalies=1, backend=backend)
    expected = {"pandas": pd.DataFrame, "polars": pl.DataFrame, "pyarrow": pa.Table}
    assert type(dataset.data) is expected[backend]
    assert dataset.events.n_events == 1


def test_labels_come_back_on_the_series_own_axis_in_the_same_flavour(
    backend: str,
) -> None:
    dataset = make_series("level_shift", n=1000, n_anomalies=3, backend=backend)
    assert type(dataset.labels) is type(dataset.data)
    # Three stretches of the default 24 samples each.
    assert labels_of(dataset).sum() == pytest.approx(3 * 24)


def test_labels_mark_exactly_the_samples_that_were_deformed() -> None:
    dataset = make_series("spike", n=1000, n_anomalies=2, strength=20.0)
    marked = np.flatnonzero(labels_of(dataset) == 1.0)
    baseline = np.median(values_of(dataset))
    assert len(marked) == 2
    assert np.all(values_of(dataset)[marked] > baseline + 10.0)


def test_the_widest_anomalies_that_fit_still_do_not_touch() -> None:
    # Two that touched would merge into one event, quietly making n_anomalies a
    # lie; the fit check exists to prevent that, so test it at its own boundary.
    for n_anomalies in (1, 2, 3, 5):
        widest = int(0.7 * 2000) // n_anomalies - 1
        dataset = make_series(
            "level_shift", n=2000, n_anomalies=n_anomalies, width=widest
        )
        assert dataset.events.n_events == n_anomalies
        assert labels_of(dataset).sum() == pytest.approx(n_anomalies * widest)
        with pytest.raises(ValueError, match="Raise n"):
            make_series(
                "level_shift", n=2000, n_anomalies=n_anomalies, width=widest + 1
            )


# ---------------------------------------------------------------------------
# make_series: each kind is the thing it claims to be
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["spike", "dip"])
def test_a_spike_or_dip_is_visible_in_the_value_distribution(kind: str) -> None:
    dataset = make_series(kind, n=2000, n_anomalies=3)
    assert recall_of(IqrDetector(), dataset) == 1.0
    assert recall_of(SpikeDetector(), dataset) == 1.0


def test_a_dip_goes_down_and_a_spike_goes_up() -> None:
    up = make_series("spike", n=1000, n_anomalies=2)
    down = make_series("dip", n=1000, n_anomalies=2)
    marked = labels_of(up) == 1.0
    assert np.all(values_of(up)[marked] > 100.0)
    assert np.all(values_of(down)[marked] < 100.0)


def test_a_level_shift_hides_from_the_distribution_but_not_from_the_mean() -> None:
    # The point of the kind: every shifted sample is well inside the normal
    # range, so a rule that judges values one at a time has nothing to see.
    dataset = make_series("level_shift", n=2000, n_anomalies=3, strength=4.0)
    assert recall_of(IqrDetector(), dataset) == 0.0
    assert recall_of(LevelShiftDetector(window=24), dataset) == 1.0


def test_a_volatility_shift_moves_the_spread_and_leaves_the_mean_alone() -> None:
    dataset = make_series(
        "volatility_shift", n=4000, n_anomalies=1, width=200, strength=8.0
    )
    values, marked = values_of(dataset), labels_of(dataset) == 1.0
    assert values[marked].mean() == pytest.approx(100.0, abs=2.0)
    assert values[~marked].mean() == pytest.approx(100.0, abs=0.2)
    assert values[marked].std() > 5.0 * values[~marked].std()


def test_a_volatility_shift_detector_finds_the_change_in_the_spread() -> None:
    # Weakened deliberately. VolatilityShiftDetector flags the *change point*,
    # and its two rolling windows straddle that point from the left, so the run
    # of flags sits almost entirely before the labelled stretch and overlaps only
    # its first few samples. thresh=1/24 asks "at least one of the event's 24
    # samples", which is the strongest claim a change-point detector supports
    # against a ground truth expressed as a stretch. At the default thresh=0.5
    # the recall is 0.0 at any strength or window.
    dataset = make_series("volatility_shift", n=2000, n_anomalies=3, strength=8.0)
    detector = VolatilityShiftDetector(window=24)
    found = found_by(detector, dataset)
    assert float(recall(dataset.events, found, 1.0 / 24)) == 1.0
    assert found.n_events == 3
    assert float(recall(dataset.events, found)) == 0.0


def test_a_seasonal_break_flattens_the_rhythm_and_holds_the_level() -> None:
    """A strength past the amplitude removes the whole cycle, leaving the noise."""
    dataset = make_series(
        "seasonal_break",
        n=2000,
        n_anomalies=3,
        period=24,
        amplitude=20.0,
        strength=25.0,
    )
    values, marked = values_of(dataset), labels_of(dataset) == 1.0
    assert values[marked].std() < 2.0
    assert values[~marked].std() > 10.0
    assert values[marked].mean() == pytest.approx(100.0, abs=1.0)


def test_the_strength_of_a_seasonal_break_decides_how_much_rhythm_is_left() -> None:
    """`strength` is a dial here as it is for every other kind, and saturates.

    It is a departure of ``strength * noise`` from what the series should have
    been, which for a rhythm of amplitude 20 means 40% of it removed at the
    default strength of 8 and all of it once strength reaches 20.
    """
    spreads = [
        values_of(
            series := make_series(
                "seasonal_break",
                n=2000,
                n_anomalies=3,
                period=24,
                amplitude=20.0,
                strength=strength,
            )
        )[labels_of(series) == 1.0].std()
        for strength in (2.0, 8.0, 25.0)
    ]
    assert spreads[0] > spreads[1] > spreads[2]
    # Saturated: asking for more than the whole cycle cannot remove more than it.
    assert spreads[2] == pytest.approx(
        values_of(
            saturated := make_series(
                "seasonal_break",
                n=2000,
                n_anomalies=3,
                period=24,
                amplitude=20.0,
                strength=100.0,
            )
        )[labels_of(saturated) == 1.0].std()
    )


def test_a_seasonal_detector_finds_the_break_in_the_rhythm() -> None:
    dataset = make_series(
        "seasonal_break", n=2000, n_anomalies=3, period=24, amplitude=20.0
    )
    assert recall_of(SeasonalDetector(period=24), dataset) == 1.0


# ---------------------------------------------------------------------------
# make_series: validation
# ---------------------------------------------------------------------------


def test_make_series_rejects_a_kind_it_does_not_know() -> None:
    with pytest.raises(KeyError, match="is not a kind make_series knows"):
        make_series("wobble")


@pytest.mark.parametrize("name", ["n", "n_anomalies", "width"])
def test_make_series_rejects_a_count_below_one(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        make_series("spike", **{name: 0})


@pytest.mark.parametrize("name", ["noise", "strength"])
@pytest.mark.parametrize("amount", [0.0, -1.5])
def test_make_series_rejects_a_non_positive_magnitude(name: str, amount: float) -> None:
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        make_series("spike", **{name: amount})


@pytest.mark.parametrize("strength", [0.5, 1.0])
def test_a_volatility_shift_needs_a_strength_above_one(strength: float) -> None:
    with pytest.raises(ValueError, match="needs strength above 1"):
        make_series("volatility_shift", strength=strength)


def test_a_seasonal_break_needs_a_period_to_break() -> None:
    with pytest.raises(ValueError, match="needs period"):
        make_series("seasonal_break")


@pytest.mark.parametrize("period", [1, 0, -3])
def test_make_series_rejects_a_period_shorter_than_two_samples(period: int) -> None:
    with pytest.raises(ValueError, match="period must be at least 2 samples"):
        make_series("spike", period=period)


def test_make_series_says_to_raise_n_when_the_anomalies_will_not_fit() -> None:
    with pytest.raises(ValueError, match="will not fit"):
        make_series("level_shift", n=100, n_anomalies=5, width=24)
    with pytest.raises(ValueError, match="Raise n"):
        make_series("level_shift", n=100, n_anomalies=5, width=24)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_a_dataset_cannot_be_reassigned() -> None:
    dataset = make_series("spike", n=200, n_anomalies=1)
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'name'"):
        dataset.name = "something else"  # type: ignore[misc]


def test_the_repr_shows_the_name_the_rows_and_the_events() -> None:
    dataset = make_series("spike", n=4032, n_anomalies=3)
    assert repr(dataset) == "Dataset('spike', 4032 rows, 3 events)"


def test_the_repr_omits_the_row_count_for_data_that_cannot_be_measured() -> None:
    events = make_series("spike", n=200, n_anomalies=3).events
    dataset = Dataset(name="hand-made", data=None, events=events, description="")
    assert repr(dataset) == "Dataset('hand-made', 3 events)"


# ---------------------------------------------------------------------------
# nab: offline, against a hand-written cache
# ---------------------------------------------------------------------------

WINDOWS: dict[str, list[list[str]]] = {
    "fakeCategory/fake_series.csv": [
        ["2024-01-01 02:00:00", "2024-01-01 04:00:00"],
        ["2024-01-01 09:00:00.000000", "2024-01-01 10:00:00.000000"],
    ],
    "fakeCategory/quiet_series.csv": [],
}
FILES = {
    "fakeCategory/fake_series.csv": nab_rows(12),
    "fakeCategory/quiet_series.csv": nab_rows(8, fractional=True),
}


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A fake NAB cache with two series in it and no network behind it."""
    write_cache(tmp_path, WINDOWS, FILES)
    return tmp_path


def test_nab_names_refuses_to_guess_when_nothing_is_cached(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="is not cached at") as caught:
        nab_names(cache_dir=tmp_path, download=False)
    assert str(tmp_path) in str(caught.value)
    assert "combined_windows.json" in str(caught.value)


def test_nab_names_lists_the_cached_catalogue_with_the_suffix_stripped(
    cache: Path,
) -> None:
    assert nab_names(cache_dir=cache, download=False) == [
        "fakeCategory/fake_series",
        "fakeCategory/quiet_series",
    ]


def test_load_nab_reads_a_cached_series_and_its_windows(cache: Path) -> None:
    dataset = load_nab("fakeCategory/fake_series", cache_dir=cache, download=False)
    assert dataset.name == "fakeCategory/fake_series"
    assert TimeSeries.from_any(dataset.data).n_rows == 12
    assert dataset.events.n_events == 2
    # 02:00-04:00 and 09:00-10:00, on an hourly axis.
    assert labels_of(dataset).tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    assert "2 labelled window(s)" in dataset.description


def test_a_trailing_csv_suffix_names_the_same_series(cache: Path) -> None:
    bare = load_nab("fakeCategory/fake_series", cache_dir=cache, download=False)
    suffixed = load_nab("fakeCategory/fake_series.csv", cache_dir=cache, download=False)
    assert bare.name == suffixed.name == "fakeCategory/fake_series"
    assert bare.events == suffixed.events


def test_a_series_with_no_labelled_windows_loads_with_no_events(cache: Path) -> None:
    dataset = load_nab("fakeCategory/quiet_series", cache_dir=cache, download=False)
    assert dataset.events.n_events == 0
    assert labels_of(dataset).tolist() == [0.0] * 8


def test_load_nab_emits_the_backend_it_was_asked_for(cache: Path) -> None:
    dataset = load_nab(
        "fakeCategory/fake_series", cache_dir=cache, download=False, backend="polars"
    )
    assert isinstance(dataset.data, pl.DataFrame)


def test_load_nab_rejects_a_name_the_catalogue_does_not_have(cache: Path) -> None:
    with pytest.raises(KeyError, match="has no series 'fakeCategory/nothing_here'"):
        load_nab("fakeCategory/nothing_here", cache_dir=cache, download=False)


def test_load_nab_suggests_a_near_miss(cache: Path) -> None:
    with pytest.raises(KeyError, match="Did you mean one of") as caught:
        load_nab("otherCategory/fake_serie", cache_dir=cache, download=False)
    assert "fakeCategory/fake_series" in str(caught.value)


def test_load_nab_rejects_a_csv_that_is_not_a_nab_series(tmp_path: Path) -> None:
    write_cache(
        tmp_path,
        {"fakeCategory/broken.csv": []},
        {"fakeCategory/broken.csv": "when,how_much\n2024-01-01 00:00:00,1.0\n"},
    )
    with pytest.raises(ValueError, match="does not look like a NAB series"):
        load_nab("fakeCategory/broken", cache_dir=tmp_path, download=False)


@pytest.mark.parametrize("fractional", [False, True])
def test_nab_timestamps_parse_with_and_without_fractional_seconds(
    tmp_path: Path, fractional: bool
) -> None:
    write_cache(
        tmp_path,
        {"fakeCategory/one.csv": []},
        {"fakeCategory/one.csv": nab_rows(6, fractional=fractional)},
    )
    ts = TimeSeries.from_any(
        load_nab("fakeCategory/one", cache_dir=tmp_path, download=False).data
    )
    assert ts.time[0] == pd.Timestamp("2024-01-01").value
    assert int(ts.time[1] - ts.time[0]) == HOUR


def test_nab_keeps_the_sub_second_part_of_a_timestamp(tmp_path: Path) -> None:
    rows = "timestamp,value\n2024-01-01 00:00:00.500000,1.0\n2024-01-01 01:00:00,2.0\n"
    write_cache(tmp_path, {"fakeCategory/one.csv": []}, {"fakeCategory/one.csv": rows})
    ts = TimeSeries.from_any(
        load_nab("fakeCategory/one", cache_dir=tmp_path, download=False).data
    )
    assert int(ts.time[0]) == pd.Timestamp("2024-01-01 00:00:00.5").value


# ---------------------------------------------------------------------------
# nab: the download path, with urlopen replaced
# ---------------------------------------------------------------------------


def test_a_download_is_cached_and_never_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: int | None = None) -> FakeResponse:
        calls.append(url)
        return FakeResponse(json.dumps(WINDOWS).encode("utf-8"))

    monkeypatch.setattr(nab_module, "urlopen", fake_urlopen)

    first = nab_names(cache_dir=tmp_path, download=True)
    assert len(calls) == 1
    assert calls[0].endswith("labels/combined_windows.json")
    assert (tmp_path / "labels" / "combined_windows.json").exists()
    # Nothing is left behind by the write-then-rename.
    assert list((tmp_path / "labels").iterdir()) == [
        tmp_path / "labels" / "combined_windows.json"
    ]

    assert nab_names(cache_dir=tmp_path, download=True) == first
    assert len(calls) == 1


def test_a_download_failure_names_the_url_it_could_not_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int | None = None) -> FakeResponse:
        raise URLError("no route to host")

    monkeypatch.setattr(nab_module, "urlopen", fake_urlopen)

    with pytest.raises(OSError, match="Could not fetch") as caught:
        nab_names(cache_dir=tmp_path, download=True)
    message = str(caught.value)
    assert re.search(r"https://\S+labels/combined_windows\.json", message)
    assert "no route to host" in message


def test_the_default_cache_follows_hazure_data_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Checked through the download=False message rather than by writing anywhere:
    # a test that fell back to the real home directory would pollute it.
    home = tmp_path / "somewhere" / "else"
    monkeypatch.setenv("HAZURE_DATA_HOME", str(home))
    with pytest.raises(OSError, match="is not cached at") as caught:
        nab_names(download=False)
    assert str(home / "nab" / "labels" / "combined_windows.json") in str(caught.value)


def test_an_empty_hazure_data_home_falls_back_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Home is redirected as well as the variable, so this cannot read — or find —
    # a cache the developer running the suite happens to have.
    monkeypatch.setenv("HAZURE_DATA_HOME", "")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(OSError, match="is not cached at") as caught:
        nab_names(download=False)
    assert str(tmp_path / ".cache" / "hazure" / "nab") in str(caught.value)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_reports_the_five_documented_numbers_per_detector() -> None:
    dataset = make_series("spike", n=1000, n_anomalies=3)
    table = compare(
        {"iqr": IqrDetector(), "spike": SpikeDetector(side="positive")}, dataset
    )
    assert set(table) == {"iqr", "spike"}
    for scores in table.values():
        assert set(scores) == COMPARE_KEYS
        assert all(isinstance(value, float) for value in scores.values())


def test_compare_separates_a_detector_that_sees_the_kind_from_one_that_cannot() -> None:
    dataset = make_series("level_shift", n=2000, n_anomalies=3, strength=4.0)
    table = compare(
        {"iqr": IqrDetector(), "shift": LevelShiftDetector(window=24)}, dataset
    )
    assert table["shift"]["recall"] == 1.0
    assert table["iqr"]["recall"] == 0.0


def test_compare_counts_the_alerts_each_detector_raised() -> None:
    dataset = make_series("spike", n=1000, n_anomalies=3)
    table = compare({"spike": SpikeDetector(side="positive")}, dataset)
    assert table["spike"]["alerts"] == 3.0
    assert table["spike"]["precision"] == 1.0
    assert table["spike"]["f1"] == 1.0


def test_fit_on_fits_each_detector_on_the_given_data_instead(cache: Path) -> None:
    dataset = make_series("spike", n=600, n_anomalies=2)
    history = load_nab("fakeCategory/fake_series", cache_dir=cache, download=False)

    on_itself = RememberFit()
    compare({"d": on_itself}, dataset)
    assert on_itself.fitted_rows == 600

    on_history = RememberFit()
    compare({"d": on_history}, dataset, fit_on=history.data)
    assert on_history.fitted_rows == 12


def test_the_delay_is_in_seconds_and_never_negative() -> None:
    dataset = make_series("level_shift", n=2000, n_anomalies=3, strength=4.0)
    table = compare(
        {
            "iqr": IqrDetector(),
            "shift": LevelShiftDetector(window=24),
            "never": ThresholdDetector(low=-1e9, high=1e9),
        },
        dataset,
    )
    for scores in table.values():
        assert np.isnan(scores["delay"]) or scores["delay"] >= 0.0
    # Whole 5-minute steps, because that is the sampling interval.
    assert table["iqr"]["delay"] == pytest.approx(1500.0)
    assert table["shift"]["delay"] == pytest.approx(0.0)
    # Nothing caught, so there is no delay to report rather than a delay of zero.
    assert table["never"]["alerts"] == 0.0
    assert np.isnan(table["never"]["delay"])


def test_compare_rejects_something_that_is_not_a_mapping() -> None:
    dataset = make_series("spike", n=200, n_anomalies=1)
    with pytest.raises(TypeError, match="needs a mapping from name to detector"):
        compare([("iqr", IqrDetector())], dataset)  # type: ignore[arg-type]


def test_compare_rejects_a_bare_scorer_and_points_at_score_detector() -> None:
    dataset = make_series("spike", n=200, n_anomalies=1)
    with pytest.raises(TypeError, match="ScoreDetector"):
        compare({"deviation": DeviationScorer()}, dataset)  # type: ignore[dict-item]


def test_compare_rejects_a_dataset_with_more_than_one_column() -> None:
    # A univariate detector fans out over the columns it is given, so the labels
    # come back two-wide and there is no single score to report.
    index = pd.date_range("2024-01-01", periods=40, freq="h", name="time")
    values = np.tile([1.0, 2.0, 3.0, 40.0], 10)
    wide = pd.DataFrame({"a": values, "b": values * 2.0}, index=index)
    dataset = Dataset(
        name="wide", data=wide, events=to_events(wide["a"] > 10.0), description=""
    )
    with pytest.raises(ValueError, match="scores one series at a time"):
        compare({"iqr": IqrDetector()}, dataset)


def test_compare_of_no_detectors_is_an_empty_table() -> None:
    assert compare({}, make_series("spike", n=200, n_anomalies=1)) == {}
