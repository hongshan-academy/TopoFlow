"""Flow-domain services shared by the HTTP layer.

Keeps server.py as a thin HTTP/streaming layer: exact-rational recovery, the
ratio-split / limit-module graph builders, and the MILP multi-solution
enumeration all live here.  Depends only on ``graph`` / ``result`` / ``solver``
and the ``builders`` / ``constructor`` packages, never on FastAPI.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from builders.merger import merge_to
from builders.splitter import find_ratio_scheme, split_leaves
from graph import Graph
from result import SolverResult
from solver import FlowModel

logger = logging.getLogger(__name__)

MAX_SOLUTIONS = 50
MAX_EXCLUSION_PATTERNS = 4096


class EdgeLike(Protocol):
    id: str


# -- Fraction approximation -------------------------------------------------
def _simplest_rational_in_interval(lo: Fraction, hi: Fraction) -> Fraction:
    """Return the rational with the smallest denominator within [lo, hi].

    Stern-Brocot / continued-fraction descent: if an integer lies in the
    interval it is the simplest answer, otherwise strip the common integer
    part and recurse on the reciprocals of the fractional remainders.
    """
    if lo > hi:
        lo, hi = hi, lo
    ceil_lo = -((-lo.numerator) // lo.denominator)
    if ceil_lo <= hi:
        return Fraction(ceil_lo)
    base = lo.numerator // lo.denominator  # floor(lo) == floor(hi) here
    inner = _simplest_rational_in_interval(1 / (hi - base), 1 / (lo - base))
    return base + 1 / inner


def _recover_fraction(value: float) -> Fraction:
    """Recover the simplest rational within 1.5 ULP of the given double.

    A correctly rounded value lies within 0.5 ULP of the true rational, so a
    solver error of one float step in either direction needs a 1.5 ULP window.
    The window is searched for the smallest-denominator rational, which removes
    floating-point noise without capping the denominator.
    """
    exact = Fraction(value)
    tol = Fraction(3, 2) * Fraction(math.ulp(value))
    return _simplest_rational_in_interval(exact - tol, exact + tol)


def approximate_fraction(value: float, tolerance: float = 1e-9) -> Dict[str, Any]:
    if not math.isfinite(value):
        return {"numerator": 0, "denominator": 1, "text": "0/1"}
    if abs(value) < tolerance:
        return {"numerator": 0, "denominator": 1, "text": "0/1"}
    if abs(value - 1) < tolerance:
        return {"numerator": 1, "denominator": 1, "text": "1/1"}
    frac = _recover_fraction(value)
    return {
        "numerator": frac.numerator,
        "denominator": frac.denominator,
        "text": f"{frac.numerator}/{frac.denominator}",
    }


# -- Ratio-split graph builder ---------------------------------------------
def layout_nodes(nodes: list[dict], edges: list[dict], exclude_to: set) -> None:
    out_nid = {n["id"] for n in nodes}
    indeg: Dict[str, int] = defaultdict(int)
    adj: Dict[str, list] = defaultdict(list)
    for e in edges:
        if e["from"] in out_nid and e["to"] in out_nid and e["to"] not in exclude_to:
            adj[e["from"]].append(e["to"])
            indeg[e["to"]] += 1

    depth: Dict[str, int] = {}
    import collections
    q: "collections.deque[str]" = collections.deque()
    for n in nodes:
        if indeg[n["id"]] == 0:
            depth[n["id"]] = 0
            q.append(n["id"])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth.get(v, -1) < depth[u] + 1:
                depth[v] = depth[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    layers: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        layers[depth.get(n["id"], 0)].append(n["id"])
    X = 150.0
    Y_START, Y_GAP = 80.0, 70.0
    y_by_id = {}
    for d, ids in layers.items():
        for j, i in enumerate(ids):
            y_by_id[i] = Y_START + j * Y_GAP
    for n in nodes:
        n["x"] = depth.get(n["id"], 0) * X
        n["y"] = y_by_id[n["id"]]


def build_ratio_graph(p: int, q: int, max_share: float | None = None) -> dict:
    a, b, N = find_ratio_scheme(p, q, max_share)
    t1, t2, t3 = N - q, p, q - p
    if min(t1, t2, t3) < 0 or p <= 0 or q <= 0:
        raise ValueError("Requires 0 < p < q with integer q")

    nodes: list[dict] = []
    edges: list[dict] = []
    counters: Dict[str, int] = defaultdict(int)

    def nid(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}_{counters[prefix]}"

    def emit(ntype: str) -> str:
        i = nid(ntype)
        nodes.append({"id": i, "type": ntype})
        return i

    in_id = emit("In")
    out1_id = emit("Out")
    out2_id = emit("Out")
    need_c1 = t1 > 0
    c1_id = emit("C") if need_c1 else None
    if need_c1:
        edges.append({"from": in_id, "to": c1_id})

    groups: Optional[Tuple[List[int], List[int], List[int]]] = None
    path: Optional[List[Tuple[int, Tuple[int, ...]]]] = None
    try:
        plan = split_leaves(N, (t1, t2, t3), max_share, max(20, N))
    except Exception:
        logger.warning("split_leaves failed; falling back to the full split tree", exc_info=True)
        plan = None
    if plan is not None:
        groups, path = plan

    class Node:
        __slots__ = ("value", "children", "group")
        def __init__(self, value):
            self.value = value
            self.children = []
            self.group = -1

    if groups is not None and path is not None:
        root = Node(N)
        leaves_nodes = [root]
        for parent_v, children_v in path:
            target = next((ln for ln in leaves_nodes if ln.value == parent_v), None)
            if target is None:
                continue
            target.children = [Node(cv) for cv in children_v]
            leaves_nodes.remove(target)
            leaves_nodes.extend(target.children)
    else:
        if max_share is not None and max_share * N <= 1:
            raise ValueError(
                f"Cannot refine splits so each share stays below {max_share:.4g} (N={N})")
        def build_full(value, ea, eb):
            if ea == 0 and eb == 0:
                return Node(value)
            n = Node(value)
            count = 3 if eb > 0 else 2
            sub = value // count
            nea, neb = (ea, eb - 1) if eb > 0 else (ea - 1, eb)
            n.children = [build_full(sub, nea, neb) for _ in range(count)]
            return n
        root = build_full(N, a, b)
        leaves_nodes = []
        def collect(n):
            if not n.children:
                leaves_nodes.append(n)
            else:
                for ch in n.children:
                    collect(ch)
        collect(root)
        groups = ([1] * t1, [1] * t2, [1] * t3)

    from collections import Counter
    remaining = [Counter(g) for g in groups]
    for ln in leaves_nodes:
        for gi in range(3):
            if remaining[gi].get(ln.value, 0) > 0:
                remaining[gi][ln.value] -= 1
                ln.group = gi
                break

    root_id: List[Optional[str]] = [None]
    leaf_ports: List[Tuple[str, int, int]] = []
    def dfs(node, parent_s):
        if node.children:
            s = emit("S")
            if parent_s:
                edges.append({"from": parent_s, "to": s})
            elif root_id[0] is None:
                root_id[0] = s
            for ch in node.children:
                dfs(ch, s)
        elif parent_s is not None:
            leaf_ports.append((parent_s, node.group, node.value))
    dfs(root, None)
    if root_id[0] is not None:
        src = c1_id if need_c1 else in_id
        edges.append({"from": src, "to": root_id[0]})

    group_ports: Dict[int, List[Tuple[str, int]]] = {0: [], 1: [], 2: []}
    for ps, g, v in leaf_ports:
        if g >= 0:
            group_ports[g].append((ps, v))
    if need_c1:
        assert c1_id is not None
        merge_to(nodes, edges, emit, group_ports[0], c1_id)
    merge_to(nodes, edges, emit, group_ports[1], out1_id)
    merge_to(nodes, edges, emit, group_ports[2], out2_id)

    layout_nodes(nodes, edges, exclude_to={c1_id} if c1_id else set())

    for i, e in enumerate(edges):
        e["id"] = f"e{i}"

    return {
        "nodes": nodes,
        "edges": edges,
        "info": {
            "p": p, "q": q, "a": a, "b": b, "N": N,
            "out1": p, "out2": q - p, "ratio": f"{p}:{q - p}",
            "feedBack": t1, "total": N,
        },
    }


# -- Limit-module graph builder --------------------------------------------
def _limit_node_type(name: str) -> str:
    if name == "In":
        return "In"
    if name == "Out":
        return "Out"
    if name.startswith(("S2_", "S3_")):
        return "S"
    if name.startswith(("C2_", "C3_")):
        return "C"
    raise ValueError(f"Unknown constructor node: {name}")


def build_limit_module(p: int, q: int, optimize: bool, search_range: int,
                       reduction_depth: int, reduction_state_limit: int,
                       cross_check: bool) -> dict:
    from constructor import boundary_flow, construct_fraction, crosscheck_karzanov

    result = construct_fraction(
        p, q,
        optimize=optimize,
        search_range=search_range,
        reduction_depth=reduction_depth,
        reduction_state_limit=reduction_state_limit,
    )
    cert = result.certificate

    node_type: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    for u, v in cert.edges:
        for name in (u, v):
            if name not in node_type:
                node_type[name] = _limit_node_type(name)
                nodes.append({"id": name, "type": node_type[name]})

    edges: List[Dict[str, Any]] = [
        {"id": f"e{i}", "from": u, "to": v}
        for i, (u, v) in enumerate(cert.edges)
    ]

    edge_flows: List[Dict[str, Any]] = []
    for i, (edge, flow, state) in enumerate(zip(cert.edges, cert.flows, cert.states)):
        edge_flows.append({
            "id": f"e{i}",
            "from": edge[0],
            "to": edge[1],
            "flow": {
                "numerator": flow.numerator,
                "denominator": flow.denominator,
                "text": f"{flow.numerator}/{flow.denominator}",
            },
            "state": state.name,
            "fixed": i in cert.fixed,
        })

    layout_nodes(nodes, edges, exclude_to=set())

    info: Dict[str, Any] = {
        "target": f"{result.target.numerator}/{result.target.denominator}",
        "flow": f"{boundary_flow(cert).numerator}/{boundary_flow(cert).denominator}",
        "strategy": result.reduction.strategy,
        "cost": {
            "nodes": result.reduction.cost.nodes,
            "edges": result.reduction.cost.edges,
            "fixedEdges": result.reduction.cost.fixed_edges,
        },
        "steps": list(result.reduction.steps()),
        "rank": result.validation.rank,
        "edgeCount": result.validation.edge_count,
        "fullRank": result.validation.full_rank,
        "valid": result.validation.valid,
        "nodeCount": cert.node_count,
        "timing": {
            "planning": result.timing.planning_seconds,
            "unitSearch": result.timing.unit_search_seconds,
            "realization": result.timing.realization_seconds,
            "validation": result.timing.validation_seconds,
            "total": result.timing.total_seconds,
        },
    }

    if cross_check:
        check = crosscheck_karzanov(cert, result.target)
        info["crossCheck"] = {
            "backend": check.backend,
            "status": check.status,
            "flow": f"{check.flow.numerator}/{check.flow.denominator}",
            "iterations": check.iterations,
            "elapsed": check.elapsed_seconds,
        }

    return {"nodes": nodes, "edges": edges, "edgeFlows": edge_flows, "info": info}


# -- MILP multi-solution enumeration ---------------------------------------
def find_free_edge_indices(result: SolverResult) -> Set[int]:
    free: Set[int] = set()
    for i, e in enumerate(result.edges):
        if e.flow >= 1.0 - 1e-8:
            free.add(i)
    return free


def generate_blocked_combos(base: List[bool], free_indices: Set[int]) -> List[List[bool]]:
    free_list = sorted(free_indices)
    if len(free_list) > MAX_EXCLUSION_PATTERNS.bit_length() - 1:
        return [list(base)]
    results = []
    for mask in range(1 << len(free_list)):
        pattern = list(base)
        for j, idx in enumerate(free_list):
            pattern[idx] = bool(mask & (1 << j))
        results.append(pattern)
    return results


def build_solution_payload(req_edges: Sequence[EdgeLike], result: SolverResult) -> Dict[str, Any]:
    edge_flows: List[Dict[str, Any]] = []
    for (req_edge, solver_edge) in zip(req_edges, result.edges):
        frac = approximate_fraction(solver_edge.flow)
        edge_flows.append({
            "id": req_edge.id,
            "flow": frac,
            "isBlocked": solver_edge.is_blocked,
        })

    node_flows_map: Dict[str, float] = {}
    for e in result.edges:
        if e.target not in node_flows_map:
            node_flows_map[e.target] = 0.0
        node_flows_map[e.target] += e.flow

    node_flows_list: List[Dict[str, Any]] = [
        {"id": nid, "flow": approximate_fraction(val)}
        for nid, val in node_flows_map.items()
    ]

    return {"edgeFlows": edge_flows, "nodeFlows": node_flows_list}


def deduplicate_solutions(solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[Tuple[Any, Any, Any], ...], Any] = {}
    for sol in solutions:
        key = tuple(
            (ef["id"], ef["flow"]["numerator"], ef["flow"]["denominator"])
            for ef in sol["edgeFlows"]
        )
        blocked = sum(1 for ef in sol["edgeFlows"] if ef["isBlocked"])
        if key not in seen or blocked < seen[key][1]:
            seen[key] = (sol, blocked)
    return [sol for sol, _ in seen.values()]


def enumerate_solutions(
    graph: Graph, req_edges: Sequence[EdgeLike], workers: int = 16
) -> tuple[List[Dict[str, Any]], bool]:
    """Run the exact MILP repeatedly, excluding known solutions until none remain."""
    solutions: List[Dict[str, Any]] = []
    pending: List[List[bool]] = []
    proved_infeasible = False
    flow = FlowModel(graph, workers)

    for i in range(MAX_SOLUTIONS):
        for pattern in pending:
            flow.add_exclusion_pattern(pattern)
        pending = []

        result = flow.solve()
        status_name = result.status

        if status_name == 'Infeasible':
            proved_infeasible = True
            break
        if status_name != 'Optimal':
            break

        payload = build_solution_payload(req_edges, result)
        solutions.append(payload)

        free = find_free_edge_indices(result)
        base = [e.is_blocked for e in result.edges]

        if free:
            combos = generate_blocked_combos(base, free)
            pending.extend(combos)
            logger.info("--- Solution %d --- (%d free edges: %s -> %d exclusion patterns)",
                        i + 1, len(free), sorted(free), len(combos))
        else:
            pending.append(base)
            logger.info("--- Solution %d ---", i + 1)

        for ef in payload["edgeFlows"]:
            logger.info("  %s : flow=%s blocked=%s", ef["id"], ef["flow"]["text"], ef["isBlocked"])
        sink_flows = {nf["id"]: nf["flow"]["text"] for nf in payload["nodeFlows"]}
        logger.info("  Node flows: %s", sink_flows)

    return solutions, proved_infeasible
