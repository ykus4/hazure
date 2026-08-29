"""Rejecting a parameter whose value is not one of a small set of names.

Several components take a parameter that is really an enumeration written as a
string — a side, a reduction, a normalisation. Python does not check those, so
each one is checked here, and by the same function: a mistyped choice should
read the same wherever it is made, and should say what was passed and what was
allowed rather than only that something was wrong.

The check is deliberately repeated in ``_build``/``_combine`` as well as in
``__init__``, because :meth:`~hazure.Configurable.set_params` assigns attributes
directly and never runs a constructor.
"""

from __future__ import annotations

__all__ = ["check_choice"]


def check_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    """Reject a parameter that is not one of a small set of names.

    Parameters
    ----------
    value
        The value to check.
    allowed
        The names it may take, in the order they should be reported.
    name
        The parameter's name, used to name it in the message.

    Raises
    ------
    ValueError
        ``value`` is not in ``allowed``.
    """
    if value not in allowed:
        msg = f"{name}={value!r} is not one of {list(allowed)}."
        raise ValueError(msg)
