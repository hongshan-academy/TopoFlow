from enum import Enum, auto
from typing import Set, List, Dict, Tuple, Optional

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
    CROSSING = auto()


STRICT_PATTERNS = DEFAULT_CONFIG.strict_patterns
PATTERNS = DEFAULT_CONFIG.patterns
EXTENDED_PATTERNS = DEFAULT_CONFIG.extended_patterns


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
    crossings: Set[Node]
    edge_pairs: Dict[Node, Tuple[Tuple[Edge, Edge], Tuple[Edge, Edge]]]

    def is_valid(self, strict: bool = False) -> bool:
        valid = STRICT_PATTERNS if strict else PATTERNS
        if self.crossings:
            return False
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
        if in_d == 2 and out_d == 2:
            return NodeType.CROSSING
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
        self.crossings = set()
        self.edge_pairs = {}
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

    def copy(self) -> 'Graph':
        new_graph = Graph.__new__(Graph)
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = self.edges.copy()
        new_graph.out_edges = defaultdict(list, {k: v.copy() for k, v in self.out_edges.items()})
        new_graph.in_edges = defaultdict(list, {k: v.copy() for k, v in self.in_edges.items()})
        new_graph.degrees = {k: v.copy() for k, v in self.degrees.items()}
        new_graph.sources = self.sources.copy()
        new_graph.sinks = self.sinks.copy()
        new_graph.splitters = self.splitters.copy()
        new_graph.convergers = self.convergers.copy()
        new_graph.isolated = self.isolated.copy()
        new_graph.crossings = self.crossings.copy()
        new_graph.edge_pairs = {
            k: ((p1[0], p1[1]), (p2[0], p2[1]))
            for k, (p1, p2) in self.edge_pairs.items()
        }
        return new_graph

    def _sync_node(self, node: Node):
        self.sources.discard(node)
        self.sinks.discard(node)
        self.splitters.discard(node)
        self.convergers.discard(node)
        self.isolated.discard(node)
        self.crossings.discard(node)
        in_d, out_d = self.degrees[node]
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
            case NodeType.CROSSING:
                self.crossings.add(node)

    def add_node(self, node: Node):
        self.nodes.add(node)
        self.degrees[node] = [0, 0]
        self.isolated.add(node)

    def remove_node(self, node: Node):
        affected_edges = list(self.in_edges[node] + self.out_edges[node])
        affected_nodes = set()
        for u, v in affected_edges:
            affected_nodes.add(u)
            affected_nodes.add(v)
            self.edges.remove((u, v))
            self.out_edges[u].remove((u, v))
            self.in_edges[v].remove((u, v))
            self.degrees[u][1] -= 1
            self.degrees[v][0] -= 1
        affected_nodes.discard(node)
        self.nodes.remove(node)
        del self.degrees[node]
        self.sources.discard(node)
        self.sinks.discard(node)
        self.splitters.discard(node)
        self.convergers.discard(node)
        self.isolated.discard(node)
        self.crossings.discard(node)
        self.edge_pairs.pop(node, None)
        for n in affected_nodes:
            self._sync_node(n)

    def add_edge(self, edge: Edge):
        u, v = edge
        if u not in self.nodes:
            self.nodes.add(u)
            self.degrees[u] = [0, 0]
            self.isolated.add(u)
        if v not in self.nodes:
            self.nodes.add(v)
            self.degrees[v] = [0, 0]
            self.isolated.add(v)
        self.edges.append(edge)
        self.out_edges[u].append(edge)
        self.in_edges[v].append(edge)
        self.degrees[u][1] += 1
        self.degrees[v][0] += 1
        self._sync_node(u)
        self._sync_node(v)

    def remove_edge(self, edge: Edge):
        u, v = edge
        self.edges.remove(edge)
        self.out_edges[u].remove(edge)
        self.in_edges[v].remove(edge)
        self.degrees[u][1] -= 1
        self.degrees[v][0] -= 1
        self._sync_node(u)
        self._sync_node(v)

    def with_added_node(self, node: Node) -> 'Graph':
        new_graph = self.copy()
        new_graph.add_node(node)
        return new_graph

    def with_removed_node(self, node: Node) -> 'Graph':
        new_graph = self.copy()
        new_graph.remove_node(node)
        return new_graph

    def with_added_edge(self, edge: Edge) -> 'Graph':
        new_graph = self.copy()
        new_graph.add_edge(edge)
        return new_graph

    def with_removed_edge(self, edge: Edge) -> 'Graph':
        new_graph = self.copy()
        new_graph.remove_edge(edge)
        return new_graph

    def contract_crossings(self) -> 'Graph':
        nodes = set(self.nodes)
        edges = list(self.edges)
        for crossing_node, (pair1, pair2) in list(self.edge_pairs.items()):
            in1, out1 = pair1
            in2, out2 = pair2
            nodes.discard(crossing_node)
            for e in (in1, out1, in2, out2):
                while e in edges:
                    edges.remove(e)
            edges.append((in1[0], out1[1]))
            edges.append((in2[0], out2[1]))
        crossing_names = {x for x in self.edge_pairs}
        edges = [e for e in edges if e[0] not in crossing_names and e[1] not in crossing_names]
        return Graph(nodes, edges)

    def underlying_graph(self) -> 'Graph':
        if not self.edge_pairs:
            return Graph(self.nodes.copy(), self.edges.copy())
        return self.contract_crossings()

    def validate_underlying(self) -> bool:
        try:
            self.underlying_graph()
            return True
        except ValueError:
            return False
