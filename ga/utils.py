import functools
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from graph import Graph, Node, Edge

_subgraph_counter = 0


def _next_tag() -> str:
    global _subgraph_counter
    _subgraph_counter += 1
    return f"__{_subgraph_counter}"


def strip_in_out(full: Graph) -> Tuple[Set[Node], List[Edge], Node, Node]:
    out_edges_from_in = full.out_edges.get("In", [])
    in_edges_to_out = full.in_edges.get("Out", [])

    if not out_edges_from_in or not in_edges_to_out:
        return set(), [], "", ""

    _, old_entry = out_edges_from_in[0]
    old_exit, _ = in_edges_to_out[0]

    tag = _next_tag()
    rename: Dict[Node, Node] = {}
    for n in full.nodes:
        if n not in ("In", "Out"):
            rename[n] = n + tag

    sub_nodes: Set[Node] = set(rename.values())
    sub_edges: List[Edge] = []
    for u, v in full.edges:
        if u == "In" or v == "Out":
            continue
        sub_edges.append((rename.get(u, u), rename.get(v, v)))

    entry = rename.get(old_entry, old_entry)
    exit_ = rename.get(old_exit, old_exit)

    return sub_nodes, sub_edges, entry, exit_


def merge_subgraph(
    g: Graph,
    sub_nodes: Set[Node],
    sub_edges: List[Edge],
    pred: Node,
    entry: Node,
    exit_: Node,
    succ: Node,
) -> Graph:
    for n in sub_nodes:
        if n not in g.nodes:
            g.add_node(n)
    for u, v in sub_edges:
        g.add_edge((u, v))
    g.add_edge((pred, entry))
    g.add_edge((exit_, succ))
    return g


@functools.lru_cache(maxsize=256)
def _find_cached(
    edges_key: Tuple[Edge, ...],
) -> Optional[Tuple[Node, Node, Set[Node], Node, Node]]:
    g = Graph.from_edges(list(edges_key))
    return _find_1in_1out_subgraph_impl(g)


def _find_1in_1out_subgraph_impl(
    g: Graph,
) -> Optional[Tuple[Node, Node, Set[Node], Node, Node]]:
    internal = [n for n in g.nodes if n not in ("In", "Out")]
    if len(internal) < 2:
        return None

    all_edges = g.edges
    candidates: List[
        Tuple[Node, Node, FrozenSet[Node], Node, Node]
    ] = []
    seen_cores: Set[FrozenSet[Node]] = set()

    # ── Stage 0: precompute full reachability (for global 1-in-1-out region) ──
    fwd_full: Dict[Node, Set[Node]] = {}
    for n in internal:
        fwd_visited: Set[Node] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in fwd_visited:
                continue
            fwd_visited.add(cur)
            for _, v in g.out_edges.get(cur, []):
                if v not in fwd_visited:
                    stack.append(v)
        fwd_visited.discard("In")
        fwd_visited.discard("Out")
        fwd_full[n] = fwd_visited

    bwd_full: Dict[Node, Set[Node]] = {}
    for n in internal:
        bwd_visited: Set[Node] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in bwd_visited:
                continue
            bwd_visited.add(cur)
            for u, _ in g.in_edges.get(cur, []):
                if u not in bwd_visited:
                    stack.append(u)
        bwd_visited.discard("In")
        bwd_visited.discard("Out")
        bwd_full[n] = bwd_visited

    def _add_candidate(R: Set[Node], e: Node, x: Node) -> None:
        if e not in R or x not in R:
            return
        entry_edges: List[Edge] = []
        exit_edges: List[Edge] = []
        for u, v in all_edges:
            if u not in R and v in R:
                entry_edges.append((u, v))
                if len(entry_edges) > 1:
                    break
            elif u in R and v not in R:
                exit_edges.append((u, v))
                if len(exit_edges) > 1:
                    break

        if len(entry_edges) == 1 and len(exit_edges) == 1:
            pred_e, entry_e = entry_edges[0]
            exit_x, succ_x = exit_edges[0]
            if entry_e == e and exit_x == x:
                core_frozen = frozenset(R)
                if core_frozen not in seen_cores:
                    seen_cores.add(core_frozen)
                    candidates.append(
                        (pred_e, e, core_frozen, x, succ_x)
                    )

    # ── Stage 1: full reachability → finds the global 1-in-1-out region ──
    for e in internal:
        for x in fwd_full[e]:
            if x == e:
                continue
            R_full = fwd_full[e] & bwd_full[x]
            R_full.discard("In")
            R_full.discard("Out")
            _add_candidate(R_full, e, x)

    # ── Stage 2: stopped BFS → finds local (inserted) 1-in-1-out regions ──
    for e in internal:
        for x in fwd_full[e]:
            if x == e:
                continue

            fwd_stop: Set[Node] = set()
            stack = [e]
            while stack:
                cur = stack.pop()
                if cur in fwd_stop:
                    continue
                fwd_stop.add(cur)
                if cur in ("In", "Out") or cur == x:
                    continue
                for _, v in g.out_edges.get(cur, []):
                    if v not in fwd_stop:
                        stack.append(v)
            fwd_stop.discard("In")
            fwd_stop.discard("Out")

            if x not in fwd_stop:
                continue

            bwd_stop: Set[Node] = set()
            stack = [x]
            while stack:
                cur = stack.pop()
                if cur in bwd_stop:
                    continue
                bwd_stop.add(cur)
                if cur in ("In", "Out") or cur == e:
                    continue
                for u, _ in g.in_edges.get(cur, []):
                    if u not in bwd_stop:
                        stack.append(u)
            bwd_stop.discard("In")
            bwd_stop.discard("Out")

            R_stop = fwd_stop & bwd_stop
            R_stop.discard("In")
            R_stop.discard("Out")
            _add_candidate(R_stop, e, x)

    if not candidates:
        return None

    all_internal_frozen = frozenset(internal)
    proper = [c for c in candidates if c[2] != all_internal_frozen]

    if proper:
        chosen = random.choice(proper)
    else:
        chosen = random.choice(candidates)

    pred, entry_node, core_frozen, exit_node, succ = chosen
    return (pred, entry_node, set(core_frozen), exit_node, succ)


def cleanup_graph(g: Graph) -> Graph:
    while True:
        changed = False

        for n in list(g.nodes):
            if n in ("In", "Out"):
                continue
            d = tuple(g.degrees.get(n, (0, 0)))
            if d[0] == 0 or d[1] == 0:
                g.remove_node(n)
                changed = True
                break

        if changed:
            continue

        for n in list(g.nodes):
            if n in ("In", "Out"):
                continue
            d = tuple(g.degrees.get(n, (0, 0)))
            if d == (1, 1):
                pred = None
                succ = None
                for u, v in g.edges:
                    if v == n:
                        pred = u
                    if u == n:
                        succ = v
                if pred is not None and succ is not None:
                    g.remove_node(n)
                    if pred != succ and (pred, succ) not in g.edges:
                        g.add_edge((pred, succ))
                    changed = True
                    break

        if changed:
            continue

        for u, v in list(g.edges):
            if u not in g.nodes or v not in g.nodes:
                g.remove_edge((u, v))
                changed = True

        if not changed:
            break

    return g


def find_1in_1out_subgraph(
    g: Graph,
) -> Optional[Tuple[Node, Node, Set[Node], Node, Node]]:
    return _find_cached(tuple(sorted(g.edges)))
