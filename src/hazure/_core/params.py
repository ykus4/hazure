"""Parameter introspection, shared by every component.

Parameters are read from the constructor signature, so there is no second list of
names to keep in step with it. A parameter that exists is therefore always
visible to ``get_params``, always carried by ``clone``, and always shown in the
repr — a component cannot be silently reconstructed with one of its settings
missing.

Components nest: a :class:`~hazure.Detector` holds a scorer and a threshold, a
:class:`~hazure.Pipeline` holds its steps. The nested parts are reachable by the
``outer__inner`` naming scikit-learn uses, so ``set_params(scorer__window=48)``
reconfigures the scorer inside a detector, and grid search over a whole model
needs nothing special. hazure components work with ``sklearn.base.clone`` without
hazure importing sklearn.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = ["Configurable"]

_C = TypeVar("_C", bound="Configurable")

#: Separates a nested part's name from the parameter inside it.
_NESTING = "__"


class Configurable:
    """Mixin giving a class ``get_params``, ``set_params``, ``clone`` and a repr.

    Subclasses must accept every parameter as a named keyword argument in
    ``__init__`` and store it on an attribute of the same name. That is the whole
    convention; nothing else needs declaring.
    """

    @classmethod
    def _parameter_names(cls) -> tuple[str, ...]:
        """Return the constructor's parameter names, in declaration order.

        A component with nothing to configure need not define ``__init__`` at
        all, and reports no parameters.

        Raises
        ------
        TypeError
            The constructor takes ``*args``, which would make parameters
            impossible to name and therefore impossible to round-trip.
        """
        if cls.__init__ is object.__init__:
            # No constructor anywhere in the hierarchy, so nothing to configure.
            # Inspecting object.__init__ would report its (*args, **kwargs).
            return ()

        signature = inspect.signature(cls.__init__)
        names: list[str] = []
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind is parameter.VAR_KEYWORD:
                continue
            if parameter.kind is parameter.VAR_POSITIONAL:
                msg = (
                    f"{cls.__name__}.__init__ takes *{name}, but hazure "
                    f"components must declare parameters by name so they can be "
                    f"inspected and reconstructed."
                )
                raise TypeError(msg)
            names.append(name)
        return tuple(names)

    def _parts(self) -> Mapping[str, Configurable]:
        """Return the nested parts reachable as ``name__parameter``.

        By default these are the parameters that are themselves configurable. A
        structure that keeps its parts inside a list — a pipeline's steps —
        overrides this to name them.

        Returns
        -------
        dict
            Part names mapped to the parts.
        """
        return {
            name: value
            for name, value in self._own_params().items()
            if isinstance(value, Configurable)
        }

    def _own_params(self) -> dict[str, Any]:
        """Return the constructor parameters alone, without nesting."""
        return {name: getattr(self, name) for name in self._parameter_names()}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return this component's parameters.

        Parameters
        ----------
        deep
            Also report the parameters of nested parts, as ``part__parameter``.

        Returns
        -------
        dict
            Parameter names mapped to their current values.

        Examples
        --------
        >>> from hazure import Detector
        >>> from hazure.scorers import DeviationScorer
        >>> from hazure.thresholds import IqrThreshold
        >>> params = Detector(DeviationScorer(), IqrThreshold()).get_params()
        >>> params["scorer__scale"], params["threshold__factor"]
        ('iqr', 3.0)
        """
        params = self._own_params()
        if deep:
            for name, part in self._parts().items():
                params.update(
                    {
                        f"{name}{_NESTING}{key}": value
                        for key, value in part.get_params().items()
                    }
                )
        return params

    def set_params(self: _C, **params: Any) -> _C:
        """Set parameters in place and return self, for chaining.

        Parameters
        ----------
        **params
            Parameters to change. ``part__parameter`` reaches into a nested part.

        Returns
        -------
        Configurable
            This component.

        Raises
        ------
        KeyError
            A name is not a parameter of this component or of the part it names.
        """
        allowed = self._parameter_names()
        parts = self._parts()
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            head, sep, rest = key.partition(_NESTING)
            if sep:
                nested.setdefault(head, {})[rest] = value
            else:
                direct[key] = value

        unknown = sorted(set(direct) - set(allowed)) + sorted(set(nested) - set(parts))
        if unknown:
            msg = (
                f"{unknown} are not parameters of {type(self).__name__}; "
                f"it accepts {list(allowed)}"
                + (f" and reaches into {list(parts)}." if parts else ".")
            )
            raise KeyError(msg)
        for name, value in direct.items():
            setattr(self, name, value)
        # Nested assignments go after direct ones, so replacing a part and then
        # configuring it in the same call configures the new part.
        parts = self._parts()
        for name, inner in nested.items():
            parts[name].set_params(**inner)
        return self

    def clone(self: _C) -> _C:
        """Return an unfitted copy carrying the same parameters.

        Nested components are cloned too, so the copy shares no fitted state
        with the original. This is what lets a univariate component fan out
        across the columns of a frame without the copies interfering.

        Returns
        -------
        Configurable
            A fresh, unfitted component.
        """
        return type(self)(
            **{name: _cloned(value) for name, value in self._own_params().items()}
        )

    def __repr__(self) -> str:
        rendered = ", ".join(f"{name}={value!r}" for name, value in self._shown())
        return f"{type(self).__name__}({rendered})"

    def _shown(self) -> Iterator[tuple[str, Any]]:
        """Yield parameters that differ from their default, for a terse repr."""
        defaults = inspect.signature(type(self).__init__).parameters
        for name, value in self._own_params().items():
            default = defaults[name].default
            if default is inspect.Parameter.empty or value != default:
                yield name, value


def _cloned(value: Any) -> Any:
    """Clone a parameter value, recursing into the containers structures use."""
    if isinstance(value, Configurable):
        return value.clone()
    if isinstance(value, list):
        return [_cloned(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cloned(item) for item in value)
    return value
