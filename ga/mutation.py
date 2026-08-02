import random
from typing import List, Optional, Set, Tuple

from graph import Graph, Node, Edge
from ga.generation import generate_strict_graph
from ga.utils import cleanup_graph, find_1in_1out_subgraph, strip_in_out, merge_subgraph


def mutate_partial_delete(g: Graph) -> Optional[Graph]:
    internal_nodes = [n for n in g.nodes if n not in ("In", "Out")]
    all_edges = list(g.edges)
    targets = internal_nodes + all_edges

    if not targets:
        return None

    target = random.choice(targets)
    g2 = g.copy()

    if isinstance(target, tuple):
        g2.remove_edge(target)
    else:
        g2.remove_node(target)

    cleanup_graph(g2)

    if "In" not in g2.nodes or "Out" not in g2.nodes:
        return None
    if not g2.is_valid(strict=True):
        return None

    return g2


def mutate_add_subgraph(g: Graph) -> Optional[Graph]:
    pairs: List[Tuple[Node, Node]] = []
    for u in g.nodes:
        if u in ("In", "Out"):
            continue
        if tuple(g.degrees.get(u, (0, 0))) != (1, 2):
            continue
        for v in g.nodes:
            if v in ("In", "Out") or u == v:
                continue
            if tuple(g.degrees.get(v, (0, 0))) != (2, 1):
                continue
            if (u, v) in g.edges:
                continue
            pairs.append((u, v))

    if not pairs:
        return None

    u, v = random.choice(pairs)
    n_nodes = len(g.nodes)
    max_k = max(n_nodes // 5, 2)
    candidate_ks = [0] + [k for k in range(2, max_k + 1) if k != 1]
    k = random.choice(candidate_ks)

    g2 = g.copy()

    if k == 0:
        g2.add_edge((u, v))
    else:
        full = generate_strict_graph(k)
        if len(full.nodes) <= 2 or len(full.edges) <= 1:
            g2.add_edge((u, v))
        else:
            sub_nodes, sub_edges, entry, exit_ = strip_in_out(full)
            if not sub_nodes:
                g2.add_edge((u, v))
            else:
                merge_subgraph(g2, sub_nodes, sub_edges, u, entry, exit_, v)

    return g2 if g2.is_valid(strict=True) else None


def mutate_replace_subgraph(g: Graph) -> Optional[Graph]:
    module = find_1in_1out_subgraph(g)

    if module is None:
        internal_edges = [(a, b) for a, b in g.edges if a != "In" and b != "Out"]
        if not internal_edges:
            return None
        u, v = random.choice(internal_edges)
        full = generate_strict_graph(2)
        if len(full.nodes) <= 2 or len(full.edges) <= 1:
            return None
        sub_nodes, sub_edges, entry, exit_ = strip_in_out(full)
        if not sub_nodes:
            return None
        g2 = g.copy()
        g2.remove_edge((u, v))
        merge_subgraph(g2, sub_nodes, sub_edges, u, entry, exit_, v)
        return g2 if g2.is_valid(strict=True) else None

    pred, entry, sub_nodes, exit_, succ = module
    k = len(sub_nodes)

    if k < 2:
        k = 2
    full = generate_strict_graph(k)
    if len(full.nodes) <= 2 or len(full.edges) <= 1:
        return None
    new_sub_nodes, new_sub_edges, new_entry, new_exit = strip_in_out(full)
    if not new_sub_nodes:
        return None

    g2 = g.copy()
    for n in sub_nodes:
        if n in g2.nodes:
            g2.remove_node(n)

    merge_subgraph(g2, new_sub_nodes, new_sub_edges, pred, new_entry, new_exit, succ)
    return g2 if g2.is_valid(strict=True) else None


def mutate_reverse_edge(g: Graph) -> Optional[Graph]:
    candidates: List[Tuple[Node, Node]] = []
    for u, v in g.edges:
        if u in ("In", "Out") or v in ("In", "Out"):
            continue
        if tuple(g.degrees.get(u, (0, 0))) == (1, 2) and tuple(g.degrees.get(v, (0, 0))) == (2, 1):
            candidates.append((u, v))

    if not candidates:
        return None

    u, v = random.choice(candidates)
    g2 = g.copy()
    g2.remove_edge((u, v))
    g2.add_edge((v, u))
    return g2 if g2.is_valid(strict=True) else None


MUTATION_FNS = [
    mutate_partial_delete,
    mutate_add_subgraph,
    mutate_replace_subgraph,
    mutate_reverse_edge,
]
