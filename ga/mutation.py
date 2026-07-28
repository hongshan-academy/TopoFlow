import random
from typing import Tuple, Optional, Callable, List

from graph import Graph, Edge, STRICT_PATTERNS
from config import DEFAULT_CONFIG as _cfg

from ga.utils import tuple_to_graph, edges_to_tuple, cascade_fix, insert_subgraph
from ga.generation import generate_subgraph


def _non_boundary_edges(graph: Graph) -> List[Edge]:
    source = next(iter(graph.sources))
    sink = next(iter(graph.sinks))
    source_out = graph.out_edges[source][0]
    sink_in = graph.in_edges[sink][0]
    return [e for e in graph.edges if e != source_out and e != sink_in]


def mutate_edge_deletion(individual: Tuple[Edge, ...]) -> Optional[Tuple[Edge, ...]]:
    graph = tuple_to_graph(individual)
    candidates = _non_boundary_edges(graph)
    if not candidates:
        return None
    edge_to_remove = random.choice(candidates)
    edges = list(graph.edges)
    edges.remove(edge_to_remove)
    nodes_set = set()
    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)
    new_graph = Graph(nodes_set, edges, _validate=False)
    if not cascade_fix(new_graph):
        return None
    return edges_to_tuple(new_graph)


def mutate_node_deletion(individual: Tuple[Edge, ...]) -> Optional[Tuple[Edge, ...]]:
    graph = tuple_to_graph(individual)
    internal = list(graph.nodes - graph.sources - graph.sinks)
    if not internal:
        return None
    node = random.choice(internal)
    edges = [(u, v) for (u, v) in graph.edges if u != node and v != node]
    nodes_set = set()
    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)
    new_graph = Graph(nodes_set, edges, _validate=False)
    if not cascade_fix(new_graph):
        return None
    return edges_to_tuple(new_graph)


def mutate_edge_addition(individual: Tuple[Edge, ...]) -> Optional[Tuple[Edge, ...]]:
    graph = tuple_to_graph(individual)
    candidates: List[Tuple[str, str]] = []
    for u in set(graph.nodes):
        if u in graph.sources or u in graph.sinks:
            continue
        in_u, out_u = graph.degrees[u]
        if (in_u, out_u) != (1, 2):
            continue
        for v in set(graph.nodes):
            if v in graph.sources or v in graph.sinks:
                continue
            if u == v:
                continue
            in_v, out_v = graph.degrees[v]
            if (in_v, out_v) != (2, 1):
                continue
            candidates.append((u, v))

    if not candidates:
        return None

    u, v = random.choice(candidates)
    graph.add_edge((u, v))
    if not graph.is_valid(strict=True):
        return None
    return edges_to_tuple(graph)


def mutate_node_addition(individual: Tuple[Edge, ...]) -> Optional[Tuple[Edge, ...]]:
    graph = tuple_to_graph(individual)
    edge = random.choice(graph.edges)
    u, v = edge
    subgraph, entry, exit_node = generate_subgraph(2)
    insert_subgraph(graph, u, v, subgraph, entry, exit_node)
    if not graph.is_valid(strict=True):
        return None
    return edges_to_tuple(graph)


def mutate_subgraph_replacement(
    individual: Tuple[Edge, ...],
) -> Optional[Tuple[Edge, ...]]:
    graph = tuple_to_graph(individual)
    edge = random.choice(graph.edges)
    u, v = edge
    n_min = _cfg.subplot_complexity_min
    n_max = _cfg.subplot_complexity_max
    n_internal = random.randint(n_min, n_max)
    if n_internal % 2 == 1:
        n_internal += 1
    subgraph, entry, exit_node = generate_subgraph(n_internal)
    insert_subgraph(graph, u, v, subgraph, entry, exit_node)
    if not graph.is_valid(strict=True):
        return None
    return edges_to_tuple(graph)


MUTATION_FNS: List[Callable] = [
    mutate_edge_deletion,
    mutate_node_deletion,
    mutate_edge_addition,
    mutate_node_addition,
    mutate_subgraph_replacement,
]
