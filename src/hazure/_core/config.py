"""Parameter introspection and state round-tripping, shared by every component.

Parameters are read from the constructor signature, so there is no second list of
names to keep in step with it. A parameter that exists is therefore always
visible to ``get_params``, always carried by ``clone``, and always shown in the
repr — a component cannot be silently reconstructed with one of its settings
missing.

The resulting contract matches scikit-learn closely enough that hazure
components work with ``sklearn.base.clone`` and grid search, without hazure
importing sklearn.

:meth:`Configurable.to_dict` and :meth:`Configurable.from_dict` extend the same
idea to *fitted* state, so a model fitted on a clean month can be stored and used
next week without refitting.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = ["Configurable"]

_C = TypeVar("_C", bound="Configurable")

#: Keys that mark an encoded value rather than a plain mapping. A dict parameter
#: using one of these would be indistinguishable from an encoded value, so
#: encoding refuses it rather than reading it back as the wrong type.
_TAGS = ("__ndarray__", "__tuple__", "__component__")

#: Where a serialised component may be imported from. A payload names a class to
#: construct, so it is a payload's only reach into the interpreter — and it
#: reaches no further than this package.
_NAMESPACE = "hazure."


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

    def to_dict(self) -> dict[str, Any]:
        """Return this component, fitted state and all, as JSON-safe data.

        Everything the instance holds is captured, not only the parameters and not
        only the public ``fitted_`` attributes: a seasonal profile is useless
        without the phase anchor it was learned against, and that anchor is
        private. Completeness is what makes the round trip safe — a component
        reconstructed from a partial payload would answer questions rather than
        raise, and answer them wrongly.

        Returns
        -------
        dict
            ``{"hazure": version, "type": import path, "state": {...}}``, holding
            only types :func:`json.dump` accepts.

        Raises
        ------
        TypeError
            Something held by this component cannot be represented — most often a
            model object handed to :class:`~hazure.OutlierScorer` or
            :class:`~hazure.MinClusterScorer`, which hazure did not build and
            cannot rebuild. Store those with :mod:`pickle`, or reconstruct the
            component and fit it again.

        See Also
        --------
        from_dict : Rebuild a component from what this returns.

        Notes
        -----
        Non-finite floats are kept as ``float("nan")`` and ``float("inf")``, which
        :mod:`json` reads and writes but which are not standard JSON. Where a
        strict parser has to consume the output, pass ``allow_nan=False`` to
        :func:`json.dumps` and it will tell you rather than emit something the
        parser will reject.

        The payload records the version that produced it for diagnosis, and
        nothing enforces it. It is a way to keep a fitted model between runs of
        the same version, not an archive format.

        Examples
        --------
        >>> import numpy as np
        >>> from hazure import TimeSeries
        >>> from hazure.scoring import DeviationScorer
        >>> time = np.arange(6, dtype=np.int64) * 3_600_000_000_000
        >>> series = TimeSeries.from_arrays(time, np.array([10, 12, 11, 13, 12, 10.0]))
        >>> stored = DeviationScorer().fit(series).to_dict()
        >>> stored["type"]
        'hazure.scoring.deviation.DeviationScorer'
        >>> stored["state"]["center_"], stored["state"]["scale_"]
        (11.5, 1.75)
        >>> DeviationScorer.from_dict(stored).center_
        11.5
        """
        return {
            "hazure": _version(),
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "state": {name: _encode(value, name) for name, value in vars(self).items()},
        }

    @classmethod
    def from_dict(cls: type[_C], payload: Mapping[str, Any]) -> _C:
        """Rebuild the component :meth:`to_dict` described.

        The class named in the payload is constructed with its parameters, then
        the remaining attributes are restored — so a component that was fitted
        comes back fitted, and one that was not comes back unfitted.

        Parameters
        ----------
        payload
            What :meth:`to_dict` returned, possibly after a trip through JSON.

        Returns
        -------
        Configurable
            The component the payload describes.

        Raises
        ------
        KeyError
            The payload has no ``"type"`` or no ``"state"``.
        TypeError
            The named class is not a hazure component, or is not the class this
            was called on — ``SpikeDetector.from_dict`` will not hand back an
            ``IqrDetector``.
        ValueError
            The payload names a class outside ``hazure``, or one that does not
            exist.

        Notes
        -----
        A payload names a class to import, so it is worth being precise about what
        that permits: the import is refused unless the name lies inside this
        package, and the result is refused unless it is a component. A payload
        from somewhere you do not trust can therefore construct a hazure
        component with odd parameters, which is a different order of problem from
        naming any importable object in the interpreter.
        """
        try:
            path, state = payload["type"], payload["state"]
        except KeyError as error:  # pragma: no cover - message is the whole point
            msg = f"a serialised component needs 'type' and 'state'; {error} missing."
            raise KeyError(msg) from error
        # Naming the class yourself is proof it is already imported, so no import
        # is needed and none is attempted. That is what lets a component of your
        # own round-trip: `MyScorer.from_dict(payload)` works, while the general
        # `Configurable.from_dict(payload)`, which has to import, stays inside
        # hazure.
        own = f"{cls.__module__}.{cls.__qualname__}"
        resolved: type[Configurable] = cls if path == own else _resolve(str(path))
        if not issubclass(resolved, cls):
            msg = (
                f"{path} is not a {cls.__name__}, so {cls.__name__}.from_dict "
                f"cannot return it."
            )
            raise TypeError(msg)
        decoded = {name: _decode(value) for name, value in dict(state).items()}
        names = resolved._parameter_names()
        component = resolved(**{n: v for n, v in decoded.items() if n in names})
        for name, value in decoded.items():
            if name not in names:
                setattr(component, name, value)
        return component

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


def _version() -> str:
    """Return the installed version, or ``"unknown"`` outside an installation."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hazure")
    except PackageNotFoundError:  # pragma: no cover - a source tree, not a wheel
        return "unknown"


def _encode(value: Any, where: str) -> Any:
    """Convert one held value into JSON-safe data.

    Parameters
    ----------
    value
        The value to encode.
    where
        Attribute name, used only to make a failure locatable.

    Returns
    -------
    object
        Data :func:`json.dump` accepts.

    Raises
    ------
    TypeError
        The value has no representation here, or a mapping uses a reserved key or
        a non-string key.
    """
    if isinstance(value, Configurable):
        return {"__component__": value.to_dict()}
    if isinstance(value, np.ndarray):
        # dtype travels with the values: read back without it, an int64 array of
        # nanosecond timestamps would return as float64 and lose the last digits.
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _encode(value.item(), where)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple):
        # Tagged, because the difference is load-bearing: a (left, right) window
        # pair is read as a pair, and a list of the same two numbers is not.
        return {"__tuple__": [_encode(item, where) for item in value]}
    if isinstance(value, list):
        return [_encode(item, where) for item in value]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                msg = (
                    f"{where} is a mapping keyed by {type(key).__name__}, which "
                    f"JSON cannot represent; hazure components must key their "
                    f"state by name."
                )
                raise TypeError(msg)
        reserved = sorted(set(value) & set(_TAGS))
        if reserved:
            msg = (
                f"{where} uses the reserved key(s) {reserved}, which would be "
                f"read back as an encoded value rather than as a mapping."
            )
            raise TypeError(msg)
        return {key: _encode(item, f"{where}[{key!r}]") for key, item in value.items()}
    msg = (
        f"{where} holds a {type(value).__name__}, which hazure cannot serialise. "
        f"Objects hazure did not build — a clustering or outlier model handed to "
        f"a scorer, say — have no reconstruction it can rely on: store the "
        f"component with pickle, or rebuild it and fit it again."
    )
    raise TypeError(msg)


def _decode(value: Any) -> Any:
    """Invert :func:`_encode`.

    Parameters
    ----------
    value
        Encoded data.

    Returns
    -------
    object
        The value it stood for.
    """
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__component__" in value:
        return Configurable.from_dict(value["__component__"])
    if "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=np.dtype(value["dtype"]))
    if "__tuple__" in value:
        return tuple(_decode(item) for item in value["__tuple__"])
    return {key: _decode(item) for key, item in value.items()}


def _resolve(path: str) -> type[Configurable]:
    """Import the component class a payload names.

    Parameters
    ----------
    path
        Dotted ``module.Class`` path, as written by :meth:`Configurable.to_dict`.

    Returns
    -------
    type
        The class.

    Raises
    ------
    TypeError
        The name resolves to something that is not a component.
    ValueError
        The name lies outside ``hazure``, or does not exist.
    """
    if not path.startswith(_NAMESPACE):
        msg = (
            f"{path!r} lies outside hazure, and a serialised component is only "
            f"ever allowed to name a class inside it."
        )
        raise ValueError(msg)
    module_name, _, class_name = path.rpartition(".")
    try:
        found = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        msg = f"{path!r} does not exist in this version of hazure."
        raise ValueError(msg) from error
    if not (isinstance(found, type) and issubclass(found, Configurable)):
        msg = f"{path!r} is not a hazure component."
        raise TypeError(msg)
    return found
