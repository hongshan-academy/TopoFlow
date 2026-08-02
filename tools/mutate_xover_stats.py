"""
Mutation & Crossover operator success-rate test on random graphs (~25 nodes).
Also verifies the new O(n^3) find_1in_1out_subgraph algorithm.
Run:  python tools/mutate_xover_stats.py
"""

import random
import sys
from typing import List, Optional, Tuple

from graph import Graph, Edge
from ga.mutation import (
    mutate_partial_delete,
    mutate_add_subgraph,
    mutate_replace_subgraph,
    mutate_reverse_edge,
)
from ga.crossover import crossover_subgraph_exchange
from ga.generation import generate_strict_graph
from ga.utils import find_1in_1out_subgraph

N_GRAPHS = 100
N_INTERNAL = 25              # internal nodes for generate_strict_graph
MUTATION_ATTEMPTS = 10       # attempts per operator per graph
CROSSOVER_PAIRS = 50         # number of random pairs to test
SEED = 42


def _edges_key(g: Graph) -> Tuple[Edge, ...]:
    return tuple(sorted(g.edges))


def _verify_find_algorithm() -> None:
    """Verify that the exact find_1in_1out_subgraph algorithm always returns
    a subgraph (never None) and often picks a proper subset."""

    print("\n" + "=" * 68)
    print("  Verifying find_1in_1out_subgraph")
    print("-" * 68)

    random.seed(SEED)
    n_ok = 0
    n_proper = 0
    n_total = 50
    n_skipped = 0

    for i in range(n_total):
        g = generate_strict_graph(N_INTERNAL)
        g2 = mutate_add_subgraph(g)
        if g2 is None:
            n_skipped += 1
            continue

        new_nodes = g2.nodes - g.nodes
        if not new_nodes:
            n_skipped += 1
            continue

        result = find_1in_1out_subgraph(g2)
        assert result is not None, f"[{i}] algorithm returned None"

        pred, entry, core, exit_node, succ = result
        internal_count = len([n for n in g2.nodes if n not in ("In", "Out")])

        if len(core) < internal_count:
            n_proper += 1
        n_ok += 1

        # Sanity: the returned entry/exit must match boundary edges
        ee = [(u, v) for u, v in g2.edges if u not in core and v in core]
        ex = [(u, v) for u, v in g2.edges if u in core and v not in core]
        assert len(ee) == 1 and len(ex) == 1, (
            f"[{i}] boundary mismatch: entry={ee} exit={ex}"
        )
        assert ee[0][1] == entry and ex[0][0] == exit_node, (
            f"[{i}] entry/exit node mismatch: got entry={entry} exit={exit_node}"
        )

    print(f"  total attempts:   {n_total}")
    print(f"  skipped:          {n_skipped}")
    print(f"  found subgraph:   {n_ok}/{n_total - n_skipped}")
    print(f"  proper subsets:   {n_proper}/{n_ok}")
    print(f"  all assertions passed: OK")
    print("-" * 68)


def main() -> None:
    random.seed(SEED)

    # ── Generate pool ──
    print(f"Generating {N_GRAPHS} random graphs (n_internal={N_INTERNAL})...")
    pool: List[Graph] = []
    for _ in range(N_GRAPHS):
        g = generate_strict_graph(N_INTERNAL)
        assert g.is_valid(strict=True), "generated graph must be strictly valid"
        pool.append(g)

    n_nodes = [len(g.nodes) for g in pool]
    n_edges = [len(g.edges) for g in pool]
    print(f"  pool ready: avg nodes={sum(n_nodes)/len(n_nodes):.1f}  "
          f"avg edges={sum(n_edges)/len(n_edges):.1f}  "
          f"nodes range=[{min(n_nodes)}, {max(n_nodes)}]  "
          f"edges range=[{min(n_edges)}, {max(n_edges)}]\n")

    # ── Mutation ──
    mut_fns = [
        ("mutate_partial_delete",     mutate_partial_delete),
        ("mutate_add_subgraph",       mutate_add_subgraph),
        ("mutate_replace_subgraph",   mutate_replace_subgraph),
        ("mutate_reverse_edge",       mutate_reverse_edge),
    ]

    results: dict = {}

    print("=" * 68)
    print(f"{'Operator':<28} {'Success':>8} {'Total':>8} {'Rate':>10}")
    print("-" * 68)

    for name, fn in mut_fns:
        ok = 0
        total = 0
        for g in pool:
            for _ in range(MUTATION_ATTEMPTS):
                result = fn(g)
                total += 1
                if result is not None:
                    ok += 1
        rate = ok / total * 100 if total else 0
        results[name] = (ok, total, rate)
        print(f"  {name:<26} {ok:>8} {total:>8} {rate:>9.1f}%")

    # ── Crossover ──
    print()
    x_ok = 0
    x_total = 0
    x_swapped = 0  # actually changed (not a copy)

    for _ in range(CROSSOVER_PAIRS):
        g1, g2 = random.sample(pool, 2)
        orig1_key = _edges_key(g1)
        orig2_key = _edges_key(g2)

        c1, c2 = crossover_subgraph_exchange(g1, g2)
        x_total += 1

        both_valid = c1.is_valid(strict=True) and c2.is_valid(strict=True)
        if both_valid:
            x_ok += 1

        c1_key = _edges_key(c1)
        c2_key = _edges_key(c2)
        both_different = (
            c1_key != orig1_key and c1_key != orig2_key
            and c2_key != orig1_key and c2_key != orig2_key
        )
        if both_different:
            x_swapped += 1

    x_rate = x_ok / x_total * 100
    swap_rate = x_swapped / x_total * 100

    name = "crossover_subgraph_exchange"
    results[name] = (x_ok, x_total, x_rate)
    print(f"  {name:<26} {x_ok:>8} {x_total:>8} {x_rate:>9.1f}%  (actually swapped: {x_swapped}/{x_total} = {swap_rate:.1f}%)")

    print("-" * 68)

    # ── Verify find_1in_1out_subgraph ──
    _verify_find_algorithm()


if __name__ == "__main__":
    main()
