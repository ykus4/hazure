"""Combining components into models.

A single detector rarely captures a real-world notion of "anomalous". More often
the answer is a small graph: extract a feature, score it, threshold it, and
require two independent signals to agree before reporting anything.

Two shapes cover almost everything:

* :class:`Pipeline` for a straight chain — the common case, and the one worth
  keeping legible.
* :class:`Graph` for anything branching or converging, expressed as named nodes
  with named inputs.

Both are themselves components, so a pipeline can be a node of a graph, and
either can be handed to anything that accepts a detector.

Wiring is checked once, when the structure is first used, and the resolved
execution order is cached until the structure changes. Intermediate results stay
as :class:`~hazure.TimeSeries` throughout, so a chain of ten steps performs one
conversion in and one out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from hazure._core.component import (
    BaseAggregator,
    BaseDetector,
    BaseScorer,
    BaseThreshold,
    BaseTransformer,
    Component,
)
from hazure._core.series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = ["Graph", "Node", "Pipeline"]

#: Reserved node name standing for the data handed to :meth:`Graph.fit`.
SOURCE = "input"

# Phrasing for error messages, so a redirect says what the structure produces
# rather than only naming a method.
_OUTPUT_DESCRIPTION: dict[str, str] = {
    "score": "continuous scores",
    "transform": "a transformed series",
    "detect": "binary labels",
}


class _Composite(Component):
    """Shared behaviour for structures whose output type depends on their parts.

    A composite answers to whichever verb its terminal component answers to.
    Calling the wrong one raises rather than quietly returning the wrong kind of
    thing, and says which verb to use instead.
    """

    trainable: ClassVar[bool] = True
    # A composite takes whatever it is given and lets each part decide: a
    # univariate step fans out over columns on its own, and a multivariate step
    # needs them all. Fanning out at the composite level would deny the second
    # kind the columns it needs.
    multivariate: ClassVar[bool] = True

    @property
    def terminal(self) -> Component | BaseAggregator:
        """The component whose output is this structure's output."""
        raise NotImplementedError

    def clone(self) -> Any:
        """Return an unfitted copy, with every contained component also cloned.

        The default implementation only clones parameters that are themselves
        components; a composite holds its parts inside a list, so it has to
        recurse itself or the copy would share fitted state with the original.

        Returns
        -------
        _Composite
            A fresh, unfitted structure.
        """
        raise NotImplementedError

    @property
    def output_kind(self) -> str:
        """Name of the verb that best describes this structure's output.

        What matters is the *kind of thing* the last component emits, not which
        class it happens to be. A structure ending in a threshold emits binary
        labels, which makes it a detector as far as a caller is concerned, so it
        reports ``"detect"`` — though ``apply()`` is accepted too, for when the
        thresholding is the point.

        Returns
        -------
        str
            One of ``"score"``, ``"transform"``, ``"detect"``.
        """
        return self._verbs()[0]

    def _verbs(self) -> tuple[str, ...]:
        """Return the accepted verbs, most descriptive first."""
        end = self.terminal
        if isinstance(end, BaseAggregator | BaseDetector):
            return ("detect",)
        if isinstance(end, BaseThreshold):
            return ("detect", "apply")
        if isinstance(end, BaseScorer):
            return ("score",)
        if isinstance(end, BaseTransformer):
            return ("transform",)
        msg = f"{type(end).__name__} is not a recognised component type."
        raise TypeError(msg)

    def _require(self, verb: str) -> None:
        """Fail with a redirect when the caller used a verb that cannot apply."""
        accepted = self._verbs()
        if verb not in accepted:
            primary = accepted[0]
            msg = (
                f"This {type(self).__name__} ends in "
                f"{type(self.terminal).__name__}, which produces "
                f"{_OUTPUT_DESCRIPTION[primary]}, so call {primary}() or "
                f"fit_{primary}() rather than {verb}()."
            )
            raise TypeError(msg)

    # Every verb is offered, and the wrong one is refused with a redirect. That
    # is friendlier than exposing only `run`, because callers reach for the verb
    # matching what they built.

    def score(self, data: Any) -> Any:
        """Score ``data``; valid when this structure ends in a scorer."""
        self._require("score")
        return self._emit(data)

    def fit_score(self, data: Any) -> Any:
        """Fit on ``data`` and score it; valid when it ends in a scorer."""
        self._require("score")
        return self._fit_emit(data)

    def apply(self, data: Any) -> Any:
        """Label ``data``; valid when this structure ends in a threshold."""
        self._require("apply")
        return self._emit(data)

    def fit_apply(self, data: Any) -> Any:
        """Fit on ``data`` and label it; valid when it ends in a threshold."""
        self._require("apply")
        return self._fit_emit(data)

    def transform(self, data: Any) -> Any:
        """Transform ``data``; valid when this structure ends in a transformer."""
        self._require("transform")
        return self._emit(data)

    def fit_transform(self, data: Any) -> Any:
        """Fit on ``data`` and transform it; valid when it ends in a transformer."""
        self._require("transform")
        return self._fit_emit(data)

    def detect(self, data: Any) -> Any:
        """Detect anomalies in ``data``; valid when it ends in a detector."""
        self._require("detect")
        return self._emit(data)

    def fit_detect(self, data: Any) -> Any:
        """Fit on ``data`` and detect anomalies in it.

        The usual entry point for unsupervised use, where one series both defines
        normal behaviour and is searched for departures from it.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        self._require("detect")
        return self._fit_emit(data)


class Pipeline(_Composite):
    """A straight chain of components, each fed by the previous one.

    Parameters
    ----------
    steps
        ``(name, component)`` pairs, executed in order. Names must be unique and
        exist so that :meth:`summary` and error messages can point at a step.

    Raises
    ------
    ValueError
        ``steps`` is empty, or a name is repeated.

    Examples
    --------
    Chain a feature, a score and a cut-off, then use it as one detector:

    >>> from hazure import Pipeline                      # doctest: +SKIP
    >>> model = Pipeline(                                # doctest: +SKIP
    ...     [
    ...         ("deseasonalise", SeasonalDecomposition(period=24)),
    ...         ("score", DeviationScorer()),
    ...         ("cut", IqrThreshold(factor=3.0)),
    ...     ]
    ... )
    >>> anomalies = model.fit_detect(series)             # doctest: +SKIP
    """

    def __init__(self, steps: Sequence[tuple[str, Component]]) -> None:
        # Stored loosely so the runtime type check below is meaningful: callers
        # reach this constructor from untyped code more often than not.
        self.steps: list[tuple[str, Any]] = list(steps)
        self._validate()

    def _validate(self) -> None:
        """Check the chain is non-empty and its names are usable."""
        if not self.steps:
            msg = "A Pipeline needs at least one step."
            raise ValueError(msg)
        names = [name for name, _ in self.steps]
        if len(set(names)) != len(names):
            repeated = sorted({n for n in names if names.count(n) > 1})
            msg = f"Step names must be unique; repeated: {repeated}."
            raise ValueError(msg)
        for name, component in self.steps:
            if not isinstance(component, Component | BaseAggregator):
                msg = (
                    f"Step {name!r} holds {type(component).__name__}, which is "
                    f"not a hazure component."
                )
                raise TypeError(msg)

    @property
    def terminal(self) -> Component | BaseAggregator:
        """The last step's component."""
        end: Component | BaseAggregator = self.steps[-1][1]
        return end

    def named_steps(self) -> dict[str, Component | BaseAggregator]:
        """Return the steps as a mapping, for reaching a fitted sub-component."""
        return dict(self.steps)

    def clone(self) -> Pipeline:
        """Return an unfitted copy whose steps are themselves fresh clones.

        Returns
        -------
        Pipeline
            A fresh, unfitted pipeline.
        """
        return Pipeline([(name, component.clone()) for name, component in self.steps])

    def _learn(self, ts: TimeSeries) -> None:
        """Fit each step on the output of the one before it."""
        self._validate()
        current = ts
        for _, component in self.steps:
            if isinstance(component, BaseAggregator):
                msg = (
                    "An aggregator takes several inputs, so it cannot sit in a "
                    "Pipeline. Use a Graph instead."
                )
                raise TypeError(msg)
            current = component.fit(current).run(current)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        current = ts
        for _, component in self.steps:
            assert isinstance(component, Component)
            current = component.run(current)
        return current

    def summary(self) -> str:
        """Return a plain-text description of the chain.

        Returns
        -------
        str
            One line per step, in execution order.
        """
        width = max(len(name) for name, _ in self.steps)
        lines = [f"Pipeline: {len(self.steps)} step(s) -> {self.output_kind}()"]
        lines += [
            f"  {i}. {name:<{width}}  {component!r}"
            for i, (name, component) in enumerate(self.steps, start=1)
        ]
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Return the chain as a Mermaid flowchart.

        Text rather than a drawing, so it renders in a README, a pull request or
        a notebook without pulling in a plotting dependency.

        Returns
        -------
        str
            Mermaid ``flowchart LR`` source.
        """
        return Graph(_chain_to_nodes(self.steps)).to_mermaid()

    def __repr__(self) -> str:
        inner = ", ".join(f"({name!r}, {c!r})" for name, c in self.steps)
        return f"Pipeline([{inner}])"


@dataclass(frozen=True, slots=True)
class Node:
    """One vertex of a :class:`Graph`.

    Parameters
    ----------
    name
        Unique label for this node. ``"input"`` is reserved for the source data.
    model
        The component to run here.
    inputs
        Names of the nodes feeding this one, or ``"input"`` for the source data.
        Several inputs are joined on the time axis before the component sees
        them; an aggregator receives them as separate labelled columns.
    columns
        Optional per-input column selection, same length as ``inputs``. ``None``
        in a position means "every column from that input". Selecting a single
        column yields a univariate series, which is what a univariate component
        expects.
    """

    name: str
    model: Component | BaseAggregator
    inputs: tuple[str, ...] = (SOURCE,)
    columns: tuple[Sequence[str] | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.columns is not None and len(self.columns) != len(self.inputs):
            msg = (
                f"Node {self.name!r} has {len(self.inputs)} input(s) but "
                f"{len(self.columns)} column selection(s); they must match."
            )
            raise ValueError(msg)


@dataclass
class _Plan:
    """A validated execution order, cached until the structure changes."""

    order: list[str] = field(default_factory=list)
    terminal: str = ""


class Graph(_Composite):
    """Components wired as a directed acyclic graph.

    Use this when the model branches or converges — scoring one series two ways
    and requiring both to agree, or feeding one transformed series to several
    detectors. For a straight chain, :class:`Pipeline` reads better.

    Parameters
    ----------
    nodes
        The vertices. Order is irrelevant; execution order is derived from the
        wiring. A mapping of ``name -> Node`` is also accepted, in which case the
        node's own ``name`` may be omitted by passing ``Node(name=key, ...)``.

    Raises
    ------
    ValueError
        Names repeat, a named input does not exist, the graph has a cycle, or
        more than one node is left unconsumed so the output would be ambiguous.

    Examples
    --------
    Require two independent signals to agree:

    >>> from hazure import Graph, Node                    # doctest: +SKIP
    >>> model = Graph(                                    # doctest: +SKIP
    ...     [
    ...         Node("spike", SpikeDetector(window=5)),
    ...         Node("shift", LevelShiftDetector(window=10)),
    ...         Node("both", AndAggregator(), inputs=("spike", "shift")),
    ...     ]
    ... )
    >>> anomalies = model.fit_detect(series)              # doctest: +SKIP
    """

    def __init__(self, nodes: Iterable[Node] | Mapping[str, Node]) -> None:
        self.nodes = list(nodes.values()) if hasattr(nodes, "values") else list(nodes)
        self._plan: _Plan | None = None
        self._resolve()

    # -- structure ----------------------------------------------------------

    def _by_name(self) -> dict[str, Node]:
        return {node.name: node for node in self.nodes}

    def _resolve(self) -> _Plan:
        """Validate the wiring and cache the execution order.

        Returns
        -------
        _Plan
            The execution order and the terminal node's name.
        """
        if self._plan is not None:
            return self._plan

        if not self.nodes:
            msg = "A Graph needs at least one node."
            raise ValueError(msg)

        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            repeated = sorted({n for n in names if names.count(n) > 1})
            msg = f"Node names must be unique; repeated: {repeated}."
            raise ValueError(msg)
        if SOURCE in names:
            msg = f"{SOURCE!r} is reserved for the source data; rename that node."
            raise ValueError(msg)

        known = set(names) | {SOURCE}
        for node in self.nodes:
            if not node.inputs:
                msg = f"Node {node.name!r} has no inputs."
                raise ValueError(msg)
            unknown = [i for i in node.inputs if i not in known]
            if unknown:
                msg = (
                    f"Node {node.name!r} names input(s) {unknown}, which do not "
                    f"exist. Available: {sorted(known)}."
                )
                raise ValueError(msg)
            if not isinstance(node.model, Component | BaseAggregator):
                msg = (
                    f"Node {node.name!r} holds {type(node.model).__name__}, "
                    f"which is not a hazure component."
                )
                raise TypeError(msg)

        order = self._topological_order()

        consumed = {i for node in self.nodes for i in node.inputs}
        unconsumed = [n for n in names if n not in consumed]
        if len(unconsumed) != 1:
            msg = (
                f"A Graph must have exactly one output, but {unconsumed} are "
                f"unconsumed. Feed the surplus into an aggregator, or drop it."
            )
            raise ValueError(msg)

        self._plan = _Plan(order=order, terminal=unconsumed[0])
        return self._plan

    def _topological_order(self) -> list[str]:
        """Order nodes so every node follows all of its inputs."""
        done = {SOURCE}
        order: list[str] = []
        remaining = list(self.nodes)
        while remaining:
            ready = [n for n in remaining if all(i in done for i in n.inputs)]
            if not ready:
                stuck = sorted(n.name for n in remaining)
                msg = (
                    f"Node(s) {stuck} can never run: their inputs form a cycle, "
                    f"or depend on each other."
                )
                raise ValueError(msg)
            for node in ready:
                done.add(node.name)
                order.append(node.name)
                remaining.remove(node)
        return order

    @property
    def terminal(self) -> Component | BaseAggregator:
        """The component whose output is the graph's output."""
        plan = self._resolve()
        return self._by_name()[plan.terminal].model

    def named_nodes(self) -> dict[str, Component | BaseAggregator]:
        """Return the nodes as a mapping, for reaching a fitted sub-component."""
        return {node.name: node.model for node in self.nodes}

    def clone(self) -> Graph:
        """Return an unfitted copy whose nodes hold themselves fresh clones.

        Returns
        -------
        Graph
            A fresh, unfitted graph.
        """
        return Graph(
            [
                Node(
                    name=node.name,
                    model=node.model.clone(),
                    inputs=node.inputs,
                    columns=node.columns,
                )
                for node in self.nodes
            ]
        )

    # -- execution ----------------------------------------------------------

    def _gather(self, node: Node, produced: dict[str, TimeSeries]) -> TimeSeries:
        """Assemble one node's input from what its predecessors produced."""
        pieces: list[TimeSeries] = []
        for position, source in enumerate(node.inputs):
            piece = produced[source]
            wanted = None if node.columns is None else node.columns[position]
            if wanted is not None:
                piece = piece.select(wanted)
            pieces.append(piece)

        if len(pieces) == 1:
            return pieces[0]

        # Two detectors commonly emit identically named columns, so qualify each
        # contribution by the node it came from before joining.
        labelled = [
            piece.wrap(piece.values, [f"{source}__{c}" for c in piece.columns])
            for source, piece in zip(node.inputs, pieces, strict=True)
        ]
        return labelled[0].join(*labelled[1:])

    def _learn(self, ts: TimeSeries) -> None:
        """Fit every node in dependency order, on its own resolved input."""
        plan = self._resolve()
        lookup = self._by_name()
        produced: dict[str, TimeSeries] = {SOURCE: ts}
        for name in plan.order:
            node = lookup[name]
            incoming = self._gather(node, produced)
            produced[name] = _fit_and_run(node, incoming)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        plan = self._resolve()
        lookup = self._by_name()
        produced: dict[str, TimeSeries] = {SOURCE: ts}
        for name in plan.order:
            node = lookup[name]
            produced[name] = _run(node, self._gather(node, produced))
        return produced[plan.terminal]

    def trace(self, data: Any) -> dict[str, Any]:
        """Run the graph and return every node's output, not just the last.

        Useful for working out which branch flagged what, or for plotting the
        intermediate series while tuning.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        dict
            Node name mapped to that node's output in native form, plus
            ``"input"`` for the data as it entered.

        Raises
        ------
        RuntimeError
            The graph has not been fitted.
        """
        if not self.fitted:
            msg = "Fit the Graph before tracing it."
            raise RuntimeError(msg)

        ts = TimeSeries.from_any(data)
        plan = self._resolve()
        lookup = self._by_name()
        produced: dict[str, TimeSeries] = {SOURCE: ts}
        for name in plan.order:
            node = lookup[name]
            produced[name] = _run(node, self._gather(node, produced))
        return {name: series.to_native() for name, series in produced.items()}

    # -- description --------------------------------------------------------

    def summary(self) -> str:
        """Return a plain-text description of the wiring, in execution order.

        Returns
        -------
        str
            One line per node.
        """
        plan = self._resolve()
        lookup = self._by_name()
        width = max(len(n) for n in plan.order)
        lines = [
            f"Graph: {len(self.nodes)} node(s) -> "
            f"{plan.terminal} -> {self.output_kind}()"
        ]
        for name in plan.order:
            node = lookup[name]
            wiring = ", ".join(node.inputs)
            lines.append(f"  {name:<{width}}  <- {wiring:<20}  {node.model!r}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Return the wiring as a Mermaid flowchart.

        Text rather than a drawing, so it renders in a README, a pull request or
        a notebook without pulling in a plotting dependency.

        Returns
        -------
        str
            Mermaid ``flowchart LR`` source.
        """
        plan = self._resolve()
        lookup = self._by_name()
        lines = ["flowchart LR", f'  {SOURCE}(["{SOURCE}"])']
        for name in plan.order:
            node = lookup[name]
            lines.append(f'  {name}["{type(node.model).__name__}<br/>{name}"]')
        for name in plan.order:
            node = lookup[name]
            for position, source in enumerate(node.inputs):
                wanted = None if node.columns is None else node.columns[position]
                label = "" if wanted is None else f"|{', '.join(wanted)}|"
                lines.append(f"  {source} -->{label} {name}")
        lines.append(f'  {plan.terminal} --> output(["output"])')
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Graph({len(self.nodes)} nodes, output={self._resolve().terminal!r})"

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)


def _fit_and_run(node: Node, incoming: TimeSeries) -> TimeSeries:
    """Fit a node's component if it can be fitted, then run it."""
    model = node.model
    if isinstance(model, BaseAggregator):
        return model._combine(incoming)
    return model.fit(incoming).run(incoming)


def _run(node: Node, incoming: TimeSeries) -> TimeSeries:
    """Run a node's component on its resolved input."""
    model = node.model
    if isinstance(model, BaseAggregator):
        return model._combine(incoming)
    return model.run(incoming)


def _chain_to_nodes(steps: Sequence[tuple[str, Component]]) -> list[Node]:
    """Express a linear chain as graph nodes, for rendering."""
    nodes: list[Node] = []
    previous = SOURCE
    for name, component in steps:
        nodes.append(Node(name=name, model=component, inputs=(previous,)))
        previous = name
    return nodes
