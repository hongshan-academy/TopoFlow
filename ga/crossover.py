import random
from typing import List, Tuple

from graph import Graph
from ga.utils import find_1in_1out_subgraph, strip_in_out


def crossover_subgraph_exchange(g1: Graph, g2: Graph) -> Tuple[Graph, Graph]:
    m1 = find_1in_1out_subgraph(g1)
    m2 = find_1in_1out_subgraph(g2)

    if m1 is None or m2 is None:
        return g1.copy(), g2.copy()

    pred1, entry1, nodes1, exit1, succ1 = m1
    pred2, entry2, nodes2, exit2, succ2 = m2

    edges1_internal: List[Tuple[str, str]] = [
        (u, v) for u, v in g1.edges if u in nodes1 and v in nodes1
    ]
    edges2_internal: List[Tuple[str, str]] = [
        (u, v) for u, v in g2.edges if u in nodes2 and v in nodes2
    ]

    g1_new = g1.copy()
    for n in nodes1:
        if n in g1_new.nodes:
            g1_new.remove_node(n)
    for n in nodes2:
        if n not in g1_new.nodes:
            g1_new.add_node(n)
    for u, v in edges2_internal:
        g1_new.add_edge((u, v))
    g1_new.add_edge((pred1, entry2))
    g1_new.add_edge((exit2, succ1))

    g2_new = g2.copy()
    for n in nodes2:
        if n in g2_new.nodes:
            g2_new.remove_node(n)
    for n in nodes1:
        if n not in g2_new.nodes:
            g2_new.add_node(n)
    for u, v in edges1_internal:
        g2_new.add_edge((u, v))
    g2_new.add_edge((pred2, entry1))
    g2_new.add_edge((exit1, succ2))

    if not g1_new.is_valid(strict=True) or not g2_new.is_valid(strict=True):
        return g1.copy(), g2.copy()

    return g1_new, g2_new


def crossover_concat(g1: Graph, g2: Graph) -> Tuple[Graph, Graph]:
    """Serial concatenation crossover.

    In → A → Out  x  In → B → Out   →   In → A → B → Out
                                          In → B → A → Out
    """
    if (
        "In" not in g1.nodes or "Out" not in g1.nodes
        or "In" not in g2.nodes or "Out" not in g2.nodes
    ):
        return g1.copy(), g2.copy()
    if not g1.out_edges.get("In") or not g1.in_edges.get("Out"):
        return g1.copy(), g2.copy()
    if not g2.out_edges.get("In") or not g2.in_edges.get("Out"):
        return g1.copy(), g2.copy()

    _, entry1 = g1.out_edges["In"][0]
    exit1, _ = g1.in_edges["Out"][0]
    _, entry2 = g2.out_edges["In"][0]
    exit2, _ = g2.in_edges["Out"][0]

    if entry1 == exit1 or entry2 == exit2:
        return g1.copy(), g2.copy()

    sub_nodes2, sub_edges2, e2, x2 = strip_in_out(g2)
    if not sub_nodes2:
        return g1.copy(), g2.copy()

    # Offspring 1: A → B
    edges1 = [(u, v) if not (u == exit1 and v == "Out") else (exit1, e2)
              for u, v in g1.edges]
    edges1.extend(sub_edges2)
    edges1.append((x2, "Out"))
    g1_new = Graph.from_edges(edges1)

    sub_nodes1, sub_edges1, e1, x1 = strip_in_out(g1)
    if not sub_nodes1:
        return g1_new if g1_new.is_valid(strict=True) else g1.copy(), g2.copy()

    # Offspring 2: B → A
    edges2 = [(u, v) if not (u == exit2 and v == "Out") else (exit2, e1)
              for u, v in g2.edges]
    edges2.extend(sub_edges1)
    edges2.append((x1, "Out"))
    g2_new = Graph.from_edges(edges2)

    if not g1_new.is_valid(strict=True) or not g2_new.is_valid(strict=True):
        return g1.copy(), g2.copy()

    return g1_new, g2_new


CROSSOVER_FNS = [
    crossover_subgraph_exchange,
    crossover_concat,
]
