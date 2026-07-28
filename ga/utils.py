from typing import Tuple, Optional

from graph import Graph, Node, Edge, STRICT_PATTERNS


def edges_to_tuple(graph: Graph) -> Tuple[Edge, ...]:
    return tuple(sorted(graph.edges))


def tuple_to_graph(edges_tuple: Tuple[Edge, ...]) -> Graph:
    edges = list(edges_tuple)
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    return Graph(nodes, edges, _validate=False)


def cascade_fix(graph: Graph) -> bool:
    edges = list(graph.edges)

    while True:
        nodes_set = set()
        for u, v in edges:
            nodes_set.add(u)
            nodes_set.add(v)

        deg = {n: [0, 0] for n in nodes_set}
        for u, v in edges:
            deg[u][1] += 1
            deg[v][0] += 1

        sources = {n for n, d in deg.items() if tuple(d) == (0, 1)}
        sinks = {n for n, d in deg.items() if tuple(d) == (1, 0)}

        found = None
        for node in nodes_set:
            d = tuple(deg[node])
            if d not in STRICT_PATTERNS:
                found = node
                break

        if found is None:
            if len(sources) > 1:
                extra = sources - {"In"}
                found = sorted(extra)[0] if extra else None
            if found is None and len(sinks) > 1:
                extra = sinks - {"Out"}
                found = sorted(extra)[0] if extra else None

        if found is None:
            break

        edges = [(u, v) for (u, v) in edges if u != found and v != found]

    final_nodes = set()
    for u, v in edges:
        final_nodes.add(u)
        final_nodes.add(v)

    if not final_nodes:
        return False

    result = Graph(final_nodes, edges, _validate=False)
    if not result.sources or not result.sinks:
        return False

    graph.nodes = result.nodes
    graph.edges = result.edges
    graph.out_edges = result.out_edges
    graph.in_edges = result.in_edges
    graph.degrees = result.degrees
    graph.sources = result.sources
    graph.sinks = result.sinks
    graph.splitters = result.splitters
    graph.convergers = result.convergers
    graph.isolated = result.isolated
    graph.crossings = result.crossings
    graph.edge_pairs = result.edge_pairs

    return graph.is_valid(strict=True)


def insert_subgraph(
    parent: Graph,
    u: Node,
    v: Node,
    subgraph: Graph,
    entry: Node,
    exit_node: Node,
):
    parent.remove_edge((u, v))
    for edge in subgraph.edges:
        parent.add_edge(edge)
    parent.add_edge((u, entry))
    parent.add_edge((exit_node, v))
