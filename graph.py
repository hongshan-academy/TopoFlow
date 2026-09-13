"""Directed-multigraph views, node classification, and topology validation."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from enum import Enum, auto
from typing import Dict, Iterable, List, Set, Tuple

from config import DEFAULT_CONFIG


Node = str
Edge = Tuple[Node, Node]


class NodeType(Enum):
    SOURCE = auto()
    SINK = auto()
    SPLITTER = auto()
    CONVERGER = auto()
    ISOLATED = auto()


class NodeKind(Enum):
    """Named TopoFlow node roles with their expected (in-degree, out-degree)."""

    IN = ("In", 0, 1)
    OUT = ("Out", 1, 0)
    S2 = ("S2_", 1, 2)
    S3 = ("S3_", 1, 3)
    C2 = ("C2_", 2, 1)
    C3 = ("C3_", 3, 1)

    def __init__(self, prefix: str, in_degree: int, out_degree: int) -> None:
        self.prefix = prefix
        self.in_degree = in_degree
        self.out_degree = out_degree

    @property
    def degrees(self) -> Tuple[int, int]:
        return self.in_degree, self.out_degree

    @classmethod
    def from_node(cls, node: Node) -> "NodeKind":
        if node == "In":
            return cls.IN
        if node == "Out":
            return cls.OUT
        match = re.fullmatch(r"(S2|S3|C2|C3)_(\d+)", node)
        if match is None:
            raise ValueError(f"unknown node name: {node!r}")
        return cls[match.group(1)]


STRICT_PATTERNS = DEFAULT_CONFIG.strict_patterns
PATTERNS = DEFAULT_CONFIG.patterns


class Graph:
    """A directed multigraph with parallel edges retained and node roles classified."""

    nodes: Set[Node]
    edges: List[Edge]
    out_edges: Dict[Node, List[Edge]]
    in_edges: Dict[Node, List[Edge]]
    degrees: Dict[Node, List[int]]

    sources: Set[Node]
    sinks: Set[Node]
    splitters: Set[Node]
    convergers: Set[Node]
    isolated: Set[Node]

    def __init__(
        self,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        _validate: bool = True,
    ) -> None:
        self.nodes = set(nodes)
        self.edges = list(edges)

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

        self.degrees = {node: [0, 0] for node in self.nodes}
        for edge in self.edges:
            source, target = edge
            if source not in self.degrees or target not in self.degrees:
                raise ValueError(f"edge {edge!r} references an unknown node")
            self.out_edges[source].append(edge)
            self.in_edges[target].append(edge)
            self.degrees[source][1] += 1
            self.degrees[target][0] += 1

        self.sources = set()
        self.sinks = set()
        self.splitters = set()
        self.convergers = set()
        self.isolated = set()
        for node in self.nodes:
            self._sync_node(node)

        if _validate and not self.is_valid():
            raise ValueError("Invalid graph")

    @classmethod
    def from_text(cls, text: str) -> "Graph":
        nodes: Set[Node] = set()
        edges: List[Edge] = []

        for line in text.splitlines():
            if "->" not in line:
                continue

            start, end = line.split("->")
            start = start.strip()
            end = end.strip()

            nodes.add(start)
            nodes.add(end)
            edges.append((start, end))

        return cls(nodes, edges)

    @classmethod
    def from_edges(cls, edges: Iterable[Edge], _validate: bool = False) -> "Graph":
        edge_tuple = tuple(edges)
        return cls(
            {node for edge in edge_tuple for node in edge},
            edge_tuple,
            _validate,
        )

    def is_valid(self, strict: bool = False) -> bool:
        valid = STRICT_PATTERNS if strict else PATTERNS
        n_source = 0
        n_sink = 0
        for in_d, out_d in self.degrees.values():
            if (in_d, out_d) not in valid:
                return False
            if in_d == 0 and out_d == 1:
                n_source += 1
            if in_d == 1 and out_d == 0:
                n_sink += 1
        if strict:
            return n_source == 1 and n_sink == 1
        return n_source >= 1 and n_sink >= 1

    def validation_errors(self) -> Tuple[str, ...]:
        """Check reserved-node naming, degrees, self-loops, and reachability."""
        errors: List[str] = []
        if "In" not in self.nodes:
            errors.append("missing reserved source node 'In'")
        if "Out" not in self.nodes:
            errors.append("missing reserved sink node 'Out'")
        for node in sorted(self.nodes):
            try:
                expected = NodeKind.from_node(node).degrees
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if tuple(self.degrees[node]) != expected:
                errors.append(
                    f"node {node!r} has degree {self.degrees[node]}, expected {expected}"
                )
        for source, target in self.edges:
            if source == target:
                errors.append(f"self-loop is not allowed: {source!r} -> {target!r}")
        if self.sources != frozenset({"In"}):
            errors.append(
                f"sources must be exactly {{'In'}}, got {sorted(self.sources)!r}"
            )
        if self.sinks != frozenset({"Out"}):
            errors.append(
                f"sinks must be exactly {{'Out'}}, got {sorted(self.sinks)!r}"
            )
        if "In" in self.nodes:
            unreachable = self.nodes - self._reachable_from("In", reverse=False)
            if unreachable:
                errors.append(f"nodes not reachable from 'In': {sorted(unreachable)!r}")
        if "Out" in self.nodes:
            dead_ends = self.nodes - self._reachable_from("Out", reverse=True)
            if dead_ends:
                errors.append(f"nodes that cannot reach 'Out': {sorted(dead_ends)!r}")
        return tuple(errors)

    def classify(self, node: Node) -> NodeType:
        in_d, out_d = self.degrees[node]
        if in_d == 0:
            if out_d == 0:
                return NodeType.ISOLATED
            return NodeType.SOURCE
        if out_d == 0:
            return NodeType.SINK
        if in_d <= out_d:
            return NodeType.SPLITTER
        return NodeType.CONVERGER

    def _reachable_from(self, start: Node, *, reverse: bool) -> Set[Node]:
        adjacency = self.in_edges if reverse else self.out_edges
        seen: Set[Node] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for source, target in adjacency.get(node, ()):
                queue.append(source if reverse else target)
        return seen

    def _sync_node(self, node: Node) -> None:
        self.sources.discard(node)
        self.sinks.discard(node)
        self.splitters.discard(node)
        self.convergers.discard(node)
        self.isolated.discard(node)
        match self.classify(node):
            case NodeType.ISOLATED:
                self.isolated.add(node)
            case NodeType.SOURCE:
                self.sources.add(node)
            case NodeType.SINK:
                self.sinks.add(node)
            case NodeType.SPLITTER:
                self.splitters.add(node)
            case NodeType.CONVERGER:
                self.convergers.add(node)
