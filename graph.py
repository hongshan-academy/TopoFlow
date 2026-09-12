from enum import Enum, auto
from typing import Set, List, Dict, Tuple

from collections import defaultdict

from config import DEFAULT_CONFIG


Node = str
Edge = Tuple[Node, Node]


class NodeType(Enum):
    SOURCE = auto()
    SINK = auto()
    SPLITTER = auto()
    CONVERGER = auto()
    ISOLATED = auto()


STRICT_PATTERNS = DEFAULT_CONFIG.strict_patterns
PATTERNS = DEFAULT_CONFIG.patterns


class Graph(object):
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

    def __init__(self, nodes: Set[Node], edges: List[Edge], _validate: bool = True) -> None:
        self.nodes = nodes
        self.edges = edges

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

        self.degrees = {node: [0, 0] for node in nodes}
        for edge in edges:
            self.out_edges[edge[0]].append(edge)
            self.in_edges[edge[1]].append(edge)
            self.degrees[edge[0]][1] += 1
            self.degrees[edge[1]][0] += 1

        self.sources = set()
        self.sinks = set()
        self.splitters = set()
        self.convergers = set()
        self.isolated = set()
        for node in nodes:
            self._sync_node(node)

        if _validate and not self.is_valid():
            raise ValueError('Invalid graph')

    @staticmethod
    def from_text(text: str) -> 'Graph':
        nodes = set()
        edges = list()

        for line in text.splitlines():
            if '->' not in line:
                continue

            start, end = line.split('->')
            start = start.strip()
            end = end.strip()

            nodes.add(start)
            nodes.add(end)
            edges.append((start, end))

        return Graph(nodes, edges)

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
