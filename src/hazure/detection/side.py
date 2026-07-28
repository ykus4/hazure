"""Which direction of departure a detector is asked to report."""

from __future__ import annotations

from typing import Final, Literal, get_args

__all__ = [
    "Side",
]


#: Which direction of excursion counts as an anomaly.
Side = Literal["both", "positive", "negative"]


_SIDES: Final = get_args(Side)


def check_side(side: object) -> None:
    """Reject a ``side`` that is not one of the three directions.

    Parameters
    ----------
    side
        The value to check.

    Raises
    ------
    ValueError
        ``side`` is not ``"both"``, ``"positive"`` or ``"negative"``.
    """
    if side not in _SIDES:
        msg = f"side={side!r} is not one of {list(_SIDES)}."
        raise ValueError(msg)
