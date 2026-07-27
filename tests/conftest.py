"""Shared fixtures.

The point of most of these is to let a single test body run against every
backend, so that "works on pandas" can never silently mean "broken on polars".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

BACKENDS = ("pandas", "polars", "pyarrow")


def make_native(
    backend: str,
    values: Sequence[float] | np.ndarray,
    *,
    start: str = "2024-01-01",
    freq: str = "h",
    name: str = "x",
    tz: str | None = None,
) -> Any:
    """Build a single-column time series in the requested backend."""
    index = pd.date_range(start, periods=len(values), freq=freq, tz=tz, name="time")
    frame = pd.DataFrame({name: np.asarray(values, dtype=float)}, index=index)
    if backend == "pandas":
        return frame
    flat = frame.reset_index()
    if backend == "polars":
        return pl.from_pandas(flat)
    if backend == "pyarrow":
        return pl.from_pandas(flat).to_arrow()
    msg = f"Unknown backend {backend!r}."
    raise ValueError(msg)


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    """Run the test once per supported dataframe backend."""
    return str(request.param)


@pytest.fixture
def native_factory(backend: str) -> Callable[..., Any]:
    """Return a ``make_native`` bound to the current backend."""

    def factory(values: Sequence[float] | np.ndarray, **kwargs: Any) -> Any:
        return make_native(backend, values, **kwargs)

    return factory


@pytest.fixture
def hourly() -> pd.Series:
    """A short, regular, hourly pandas Series."""
    index = pd.date_range("2024-01-01", periods=24, freq="h", name="time")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(size=24), index=index, name="x")
