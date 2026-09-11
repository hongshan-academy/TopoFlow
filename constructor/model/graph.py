"""Immutable directed-multigraph views and topology validation."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable
from enum import Enum, auto

type Node = str
type Edge = tuple[Node, Node]


class NodeType(Enum):
    SOURCE = auto()
    SINK = auto()
    SPLITTER = auto()
    CONVERGER = auto()
    ISOLATED = auto()


class NodeKind(Enum):
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
    def degrees(self) -> tuple[int, int]:
        return self.in_degree, self.out_degree

    @classmethod
    def from_node(cls, node: Node) -> NodeKind:
        if node == "In":
            return cls.IN
        if node == "Out":
            return cls.OUT
        match = re.fullmatch(r"(S2|S3|C2|C3)_(\d+)", node)
        if match is None:
            raise ValueError(f"unknown node name: {node!r}")
        return cls[match.group(1)]


class Graph:
    """A completed TopoFlow directed multigraph; parallel edges are retained."""

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        self.nodes = frozenset(nodes)
        self.edges = tuple(edges)
        out_edges: defaultdict[Node, list[Edge]] = defaultdict(list)
        in_edges: defaultdict[Node, list[Edge]] = defaultdict(list)
        degrees = {node: [0, 0] for node in self.nodes}
        for edge in self.edges:
            source, target = edge
            if source not in degrees or target not in degrees:
                raise ValueError(f"edge {edge!r} references an unknown node")
            out_edges[source].append(edge)
            in_edges[target].append(edge)
            degrees[source][1] += 1
            degrees[target][0] += 1
        self.out_edges = {node: tuple(out_edges[node]) for node in self.nodes}
        self.in_edges = {node: tuple(in_edges[node]) for node in self.nodes}
        self.degrees = {node: tuple(value) for node, value in degrees.items()}
        classified: dict[NodeType, set[Node]] = {kind: set() for kind in NodeType}
        for node in self.nodes:
            classified[self.classify(node)].add(node)
        self.sources = frozenset(classified[NodeType.SOURCE])
        self.sinks = frozenset(classified[NodeType.SINK])

    @classmethod
    def from_edges(cls, edges: Iterable[Edge]) -> Graph:
        edge_tuple = tuple(edges)
        return cls({node for edge in edge_tuple for node in edge}, edge_tuple)

    def classify(self, node: Node) -> NodeType:
        in_degree, out_degree = self.degrees[node]
        if in_degree == 0:
            return NodeType.ISOLATED if out_degree == 0 else NodeType.SOURCE
        if out_degree == 0:
            return NodeType.SINK
        return NodeType.SPLITTER if in_degree <= out_degree else NodeType.CONVERGER

    def is_valid(self) -> bool:
        return not self.validation_errors()

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
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
            if self.degrees[node] != expected:
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

    def _reachable_from(self, start: Node, *, reverse: bool) -> frozenset[Node]:
        adjacency = self.in_edges if reverse else self.out_edges
        seen: set[Node] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for source, target in adjacency[node]:
                queue.append(source if reverse else target)
        return frozenset(seen)
