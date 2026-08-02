from collections import defaultdict
from typing import Dict, List, Tuple

from graph import Graph, Edge

FEATURE_NAMES = [
    "n_nodes", "n_edges", "n_internal",
    "count_12", "count_13", "count_21", "count_31",
    "avg_in_deg", "avg_out_deg", "max_in_deg", "max_out_deg",
    "edge_density", "ratio_12_to_21", "ratio_13_to_31",
    "longest_path_norm", "is_valid",
]

FEATURE_DIM = len(FEATURE_NAMES)


def _longest_path_from_in(g: Graph) -> int:
    in_degree: Dict[str, int] = defaultdict(int)
    for u in g.nodes:
        in_degree[u] = len(g.in_edges.get(u, []))
    dist: Dict[str, int] = {n: 0 for n in g.nodes}
    queue = ["In"]
    while queue:
        u = queue.pop(0)
        for _, v in g.out_edges.get(u, []):
            dist[v] = max(dist[v], dist[u] + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return dist.get("Out", 0)


def extract_features(edges_tuple: Tuple[Edge, ...]) -> List[float]:
    g = Graph.from_edges(list(edges_tuple))

    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    n_internal = n_nodes - 2

    deg_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    in_degs: List[float] = []
    out_degs: List[float] = []
    internal_nodes = [n for n in g.nodes if n not in ("In", "Out")]

    for n in internal_nodes:
        d = tuple(g.degrees.get(n, (0, 0)))
        deg_counts[d] += 1 # type: ignore
        in_degs.append(float(d[0]))
        out_degs.append(float(d[1]))

    count_12 = float(deg_counts.get((1, 2), 0))
    count_13 = float(deg_counts.get((1, 3), 0))
    count_21 = float(deg_counts.get((2, 1), 0))
    count_31 = float(deg_counts.get((3, 1), 0))

    avg_in = sum(in_degs) / len(in_degs) if in_degs else 0.0
    avg_out = sum(out_degs) / len(out_degs) if out_degs else 0.0
    max_in = (max(in_degs) if in_degs else 0) / 3.0
    max_out = (max(out_degs) if out_degs else 0) / 3.0

    max_possible = n_nodes * (n_nodes - 1) / 2.0
    edge_density = n_edges / max_possible if max_possible > 0 else 0.0

    ratio_12_to_21 = count_12 / max(count_21, 1.0)
    ratio_13_to_31 = count_13 / max(count_31, 1.0)

    longest_path = _longest_path_from_in(g)
    longest_path_norm = longest_path / max(n_internal, 1)

    is_valid = 1.0 if g.is_valid() else 0.0

    return [
        float(n_nodes),
        float(n_edges),
        float(n_internal),
        count_12, count_13, count_21, count_31,
        avg_in / 3.0,
        avg_out / 3.0,
        max_in,
        max_out,
        edge_density,
        ratio_12_to_21,
        ratio_13_to_31,
        longest_path_norm,
        is_valid,
    ]
