from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hazure.events import Events  # noqa: E402
from hazure.plotting import plot  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close whatever a test drew, so figures never accumulate."""
    yield
    plt.close("all")


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two hourly columns, with a planted spike at position 20 of column ``a``."""
    index = pd.date_range("2024-01-01", periods=48, freq="h", name="time")
    a = np.sin(np.arange(48) / 6.0)
    a[20] += 8.0
    return pd.DataFrame({"a": a, "b": np.cos(np.arange(48) / 6.0)}, index=index)


@pytest.fixture
def labels(frame: pd.DataFrame) -> pd.Series:
    """Labels flagging the planted spike."""
    values = np.zeros(len(frame))
    values[20] = 1.0
    return pd.Series(values, index=frame.index, name="a")


def test_a_series_alone_gives_back_a_figure_and_one_axes(frame: pd.DataFrame) -> None:
    fig, axes = plot(frame["a"])

    assert isinstance(fig, Figure)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)
    assert axes[0] in fig.axes
    (line,) = axes[0].get_lines()
    assert len(line.get_xydata()) == len(frame)
    assert axes[0].get_xlabel() == "time"


def test_labels_shade_the_flagged_interval(
    frame: pd.DataFrame, labels: pd.Series
) -> None:
    _, axes = plot(frame["a"], anomaly=labels)

    # One patch for the one flagged hour, plus a legend naming line and overlay.
    assert len(axes[0].patches) == 1
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == ["a", "anomaly"]


def test_events_are_accepted_in_place_of_labels(frame: pd.DataFrame) -> None:
    # 05:00 to 09:00 inclusive is five hourly samples.
    events = Events.from_any([("2024-01-01T05:00", "2024-01-01T09:00")])

    _, shaded = plot(frame["a"], anomaly=events)
    _, marked = plot(frame["a"], anomaly=events, style="marker")

    assert len(shaded[0].patches) == 1
    (points,) = [ln for ln in marked[0].get_lines() if ln.get_label() == "anomaly"]
    assert points.get_ydata() == pytest.approx(frame["a"].to_numpy()[5:10])


def test_a_mapping_of_detectors_gets_one_colour_and_one_legend_entry_each(
    frame: pd.DataFrame, labels: pd.Series
) -> None:
    other = pd.Series(np.zeros(len(frame)), index=frame.index)
    other.iloc[30:33] = 1.0

    _, axes = plot(frame["a"], anomaly={"spike": labels, "shift": other})

    texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert texts == ["a", "spike", "shift"]
    assert len({to_hex(p.get_facecolor()) for p in axes[0].patches}) == 2


def test_a_score_is_drawn_on_its_own_panel_below_the_data(
    frame: pd.DataFrame,
) -> None:
    score = pd.Series(np.arange(len(frame), dtype=float) * 1000.0, index=frame.index)

    _, axes = plot(frame["a"], score=score)

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "a"
    # The score keeps its own scale rather than flattening the data it describes.
    assert axes[1].get_ylim()[1] > 1000.0
    assert axes[0].get_ylim()[1] < 100.0


def test_layout_controls_how_columns_share_panels(frame: pd.DataFrame) -> None:
    _, each = plot(frame)
    assert len(each) == 2
    assert [ln.get_label() for ln in each[0].get_lines()] == ["a"]

    _, shared = plot(frame, layout="all")
    assert len(shared) == 1
    assert [ln.get_label() for ln in shared[0].get_lines()] == ["a", "b"]

    _, grouped = plot(frame, layout=[["a", "b"], "b"])
    assert [len(ax.get_lines()) for ax in grouped] == [2, 1]


def test_style_marker_marks_points_instead_of_shading_spans(
    frame: pd.DataFrame, labels: pd.Series
) -> None:
    _, spans = plot(frame["a"], anomaly=labels, style="span")
    _, markers = plot(frame["a"], anomaly=labels, style="marker")

    assert len(spans[0].patches) == 1
    assert not markers[0].patches
    (marked,) = [ln for ln in markers[0].get_lines() if ln.get_label() == "anomaly"]
    assert marked.get_ydata() == pytest.approx([frame["a"].iloc[20]])


def test_a_palette_overrides_the_anomaly_colours(
    frame: pd.DataFrame, labels: pd.Series
) -> None:
    named = {"one": labels, "two": labels}

    _, axes = plot(frame["a"], anomaly=named, palette={"one": "red", "two": "blue"})
    assert {to_hex(p.get_facecolor()) for p in axes[0].patches} == {
        "#ff0000",
        "#0000ff",
    }

    with pytest.raises(ValueError, match="no colour for"):
        plot(frame["a"], anomaly=named, palette={"one": "red"})


def test_supplied_axes_are_drawn_into_and_the_surplus_left_alone(
    frame: pd.DataFrame,
) -> None:
    figure, supplied = plt.subplots(3, 1)

    fig, axes = plot(frame, axes=supplied, title="two of three")

    assert fig is figure
    assert axes == list(supplied[:2])
    assert not supplied[2].get_lines()

    with pytest.raises(ValueError, match="needs 2 axes"):
        plot(frame, axes=supplied[:1])


def test_plot_leaves_the_global_rcparams_untouched(
    frame: pd.DataFrame, labels: pd.Series
) -> None:
    before = plt.rcParams.copy()

    plot(
        frame,
        anomaly={"spike": labels},
        score=pd.Series(np.ones(len(frame)), index=frame.index),
        style="marker",
        title="everything at once",
    )

    after = plt.rcParams
    assert set(after) == set(before)
    changed = [key for key in before if repr(after[key]) != repr(before[key])]
    assert changed == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"style": "shade"}, "must be one of"),
        ({"alpha": 1.5}, r"must lie in \[0, 1\]"),
        ({"layout": "per-column"}, "must be"),
        ({"layout": ["a", "missing"]}, "unknown column"),
    ],
)
def test_bad_arguments_are_refused_with_an_actionable_message(
    frame: pd.DataFrame, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plot(frame["a"], **kwargs)


def test_plot_needs_something_to_draw() -> None:
    with pytest.raises(ValueError, match="at least one of"):
        plot()
