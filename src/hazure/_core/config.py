"""Parameter introspection shared by every component.

Parameters are read from the constructor signature, so there is no second list of
names to keep in step with it. A parameter that exists is therefore always
visible to ``get_params``, always carried by ``clone``, and always shown in the
repr — a component cannot be silently reconstructed with one of its settings
missing.

The resulting contract matches scikit-learn closely enough that hazure
components work with ``sklearn.base.clone`` and grid search, without hazure
importing sklearn.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Configurable"]

_C = TypeVar("_C", bound="Configurable")


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

    def get_params(self) -> dict[str, Any]:
        """Return this component's parameters.

        Returns
        -------
        dict
            Parameter names mapped to their current values.
        """
        return {name: getattr(self, name) for name in self._parameter_names()}

    def set_params(self: _C, **params: Any) -> _C:
        """Set parameters in place and return self, for chaining.

        Parameters
        ----------
        **params
            Parameters to change.

        Returns
        -------
        Configurable
            This component.

        Raises
        ------
        KeyError
            A name is not a parameter of this component.
        """
        allowed = self._parameter_names()
        unknown = sorted(set(params) - set(allowed))
        if unknown:
            msg = (
                f"{unknown} are not parameters of {type(self).__name__}; "
                f"it accepts {list(allowed)}."
            )
            raise KeyError(msg)
        for name, value in params.items():
            setattr(self, name, value)
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
        params = {
            name: value.clone() if isinstance(value, Configurable) else value
            for name, value in self.get_params().items()
        }
        return type(self)(**params)

    def __repr__(self) -> str:
        rendered = ", ".join(f"{name}={value!r}" for name, value in self._shown())
        return f"{type(self).__name__}({rendered})"

    def _shown(self) -> Iterator[tuple[str, Any]]:
        """Yield parameters that differ from their default, for a terse repr."""
        defaults = inspect.signature(type(self).__init__).parameters
        for name, value in self.get_params().items():
            default = defaults[name].default
            if default is inspect.Parameter.empty or value != default:
                yield name, value
