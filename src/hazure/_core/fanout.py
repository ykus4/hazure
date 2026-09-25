"""Running a per-series component over every column of a frame.

Most algorithms judge one series at a time. Handed a frame, such a component
trains one independent copy of itself per column, so that each column learns its
own notion of normal, and at run time sends every column to the copy that
learned it. Nothing here knows what the component computes; it only routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hazure._core.component import Component
    from hazure._core.series import TimeSeries

__all__ = ["check_columns", "fit_columns", "join_all", "run_columns"]


def fit_columns(template: Component, ts: TimeSeries) -> dict[str, Component]:
    """Train one fresh copy of ``template`` on each column of ``ts``.

    Parameters
    ----------
    template
        The component to copy. It is cloned, never fitted itself.
    ts
        Training data with one or more columns.

    Returns
    -------
    dict
        Column names mapped to the copy fitted on that column.
    """
    models = {name: template.clone() for name in ts.columns}
    for name, model in models.items():
        model.fit(ts.select(name))
    return models


def run_columns(
    template: Component, models: dict[str, Component] | None, ts: TimeSeries
) -> TimeSeries:
    """Apply each column's copy to that column, and join the results.

    Parameters
    ----------
    template
        Used for every column when ``models`` is None — an untrainable component
        has nothing column-specific to hold.
    models
        What :func:`fit_columns` returned, or None.
    ts
        Input with one or more columns.

    Returns
    -------
    TimeSeries
        One output column per input column, or several where the component
        widens a column; those are qualified as ``column_output``.
    """
    parts = []
    for name in ts.columns:
        model = template if models is None else models[name]
        result = model._compute(ts.select(name))
        if result.columns != (name,):
            # A component that widens one column into several (lagging, say)
            # would otherwise collide across columns, so qualify the names.
            result = result.wrap(
                result.values, [f"{name}_{column}" for column in result.columns]
            )
        parts.append(result)
    return join_all(parts)


def check_columns(component: Component, ts: TimeSeries) -> TimeSeries:
    """Reconcile an input's columns with the ones training saw.

    Parameters
    ----------
    component
        A fitted component.
    ts
        The input about to be run.

    Returns
    -------
    TimeSeries
        ``ts``, reordered to the training layout for a multivariate component.

    Raises
    ------
    ValueError
        A column has no trained copy, or a multivariate component is missing a
        column it was trained on.
    """
    learned = component.feature_names
    if learned is None:
        return ts
    models = component._column_models
    name = type(component).__name__

    if models is not None:
        missing = [c for c in ts.columns if c not in models]
        if missing:
            msg = (
                f"{name} was fitted on {list(learned)} and has nothing trained "
                f"for {missing}."
            )
            raise ValueError(msg)
        return ts

    if component.is_multivariate:
        missing = [c for c in learned if c not in ts.columns]
        if missing:
            msg = (
                f"{name} was fitted on {list(learned)} but the input is missing "
                f"{missing}."
            )
            raise ValueError(msg)
        # Reorder to the training layout; extra columns are dropped, since the
        # model has no coefficients for them.
        return ts.select(learned)
    return ts


def join_all(parts: Iterable[TimeSeries]) -> TimeSeries:
    """Join series into one, aligning on the time axis."""
    materialised = list(parts)
    if not materialised:  # pragma: no cover - callers always pass at least one
        msg = "Nothing to combine."
        raise ValueError(msg)
    first, *rest = materialised
    return first.join(*rest)
