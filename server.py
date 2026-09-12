"""Unified TopoFlow web backend (FastAPI).

Endpoints
  - GET  /api/config          discrete simulation config
  - GET  /api/solvers         list of available solvers
  - POST /api/solve           flow solving (engine = milp / rank-smt / stable)
  - POST /api/simulate        discrete-event simulation (frame replay)
  - POST /api/ratio-split     standard ratio-split module builder
  - POST /api/limit-module    standard limit-flow computation (constructor)
  - POST /api/topoflow-layout physical layout (layout / Z3 bridge, NDJSON stream)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from solver import solve, solve_native
from simulator import simulate_frames
from graph import Graph
from config import DEFAULT_CONFIG
from result import SolverResult

from builders.splitter import find_ratio_scheme, split_leaves, bfs_min_splits
from builders.merger import merge_to

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("topoflow")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

app = FastAPI(title="TopoFlow API")


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


# -- Request models ---------------------------------------------------------
class NodeModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(validation_alias=AliasChoices("id", "node_id"))
    type: str = Field(default="", validation_alias=AliasChoices("type", "node_type"))
    x: float = 0
    y: float = 0


class EdgeModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_: str = Field(alias="from")
    to: str


class SolveRequest(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    engine: str = "milp"


class SimulateOptions(BaseModel):
    max_frames: Optional[int] = None


class SimulateRequest(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    options: SimulateOptions = Field(default_factory=SimulateOptions)


class LimitModuleRequest(BaseModel):
    p: int
    q: int
    optimize: bool = False
    search_range: int = 0
    reduction_depth: int = 3
    reduction_state_limit: int = 4096
    cross_check: bool = False


MAX_SOLUTIONS = 50


def _find_free_edge_indices(result: SolverResult) -> Set[int]:
    free: Set[int] = set()
    for i, e in enumerate(result.edges):
        if e.flow >= 1.0 - 1e-8:
            free.add(i)
    return free


def _generate_blocked_combos(base: List[bool], free_indices: Set[int]) -> List[List[bool]]:
    free_list = sorted(free_indices)
    results = []
    for mask in range(1 << len(free_list)):
        pattern = list(base)
        for j, idx in enumerate(free_list):
            pattern[idx] = bool(mask & (1 << j))
        results.append(pattern)
    return results


def _build_solution_payload(req_edges: List[EdgeModel], result: SolverResult) -> Dict[str, Any]:
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


def _deduplicate_solutions(solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# -- Common helpers ---------------------------------------------------------
def _raw_graph_from_body(body: dict) -> tuple[list[dict], list[dict]]:
    nodes = body.get("nodes") or body.get("rawNodes", [])
    edges = body.get("edges") or body.get("rawEdges", [])
    return nodes, edges


# -- Physical layout bridge (layout package, exact Z3 bridge) ---------------
def _get_layout_adapter():
    from layout import adapter

    return adapter


def _convert_for_topoflow(nodes: list[dict], edges: list[dict]) -> dict:
    in_deg: Dict[str, int] = defaultdict(int)
    out_deg: Dict[str, int] = defaultdict(int)
    for e in edges:
        out_deg[e["from"]] += 1
        in_deg[e["to"]] += 1

    id_mapping: dict[str, str] = {}
    type_counters: Dict[str, int] = defaultdict(int)
    for n in nodes:
        nid, ntype = n["id"], n["type"]
        if ntype == "In":
            id_mapping[nid] = "In"
            continue
        if ntype == "Out":
            id_mapping[nid] = "Out"
            continue
        out_d, in_d = out_deg.get(nid, 0), in_deg.get(nid, 0)
        if ntype == "S":
            if out_d < 2 or out_d > 3:
                raise ValueError(f"分流器 {nid} 出度 {out_d} 不在 2~3，无法物理求解")
            tf_type = "S3" if out_d == 3 else "S2"
        elif ntype == "C":
            if in_d < 2 or in_d > 3:
                raise ValueError(f"汇流器 {nid} 入度 {in_d} 不在 2~3，无法物理求解")
            tf_type = "C3" if in_d == 3 else "C2"
        else:
            raise ValueError(f"不支持的节点类型: {ntype}")
        id_mapping[nid] = f"{tf_type}_{type_counters[tf_type]}"
        type_counters[tf_type] += 1

    return {
        "nodes": [id_mapping[n["id"]] for n in nodes],
        "edges": [[id_mapping[e["from"]], id_mapping[e["to"]]] for e in edges],
    }


def _try_solve(adapter, topo_graph: dict, rows: int, cols: int,
               require_cell: bool, time_limit: float) -> tuple[dict, int]:
    mid = rows // 2
    adapted = adapter.adapt_topoflow(
        topo_graph,
        rank=1,
        rows=rows,
        columns=cols,
        transport="belt",
        require_splitter_merger_belt_cell=require_cell,
        terminal_anchors={
            "In": {"side": "left", "offset": mid},
            "Out": {"side": "right", "offset": mid},
        },
    )
    solution, _label, route_length = adapter.solve_topoflow(
        adapted,
        time_limit=time_limit,
        workers=4,
        search_mode="balanced",
    )
    return solution, route_length


def _find_min_grid(adapter, topo_graph: dict, require_cell: bool,
                   time_limit: float, min_size: int,
                   progress_cb=None) -> tuple[int, int, dict, int] | None:
    n_internal = sum(1 for nid in topo_graph["nodes"]
                     if nid not in ("In", "Out"))
    start = max(min_size, n_internal + 2)

    size = start
    cur_r = cur_c = size
    sol = route_length = None
    cap = 30
    while True:
        if progress_cb:
            progress_cb(size, size)
        try:
            sol, route_length = _try_solve(adapter, topo_graph, size, size,
                                           require_cell, time_limit)
            cur_r = cur_c = size
            break
        except adapter.SolveFailure:
            if size < cap:
                size += 1
            else:
                cap = size
                size *= 2
    if sol is None:
        return None
    cur_sol, cur_rl = sol, route_length

    while cur_r > min_size or cur_c > min_size:
        candidates = []
        if cur_r > min_size and cur_c > min_size:
            candidates.append((cur_r - 1, cur_c - 1))
        if cur_c > min_size:
            candidates.append((cur_r, cur_c - 1))
        if cur_r > min_size:
            candidates.append((cur_r - 1, cur_c))

        compressed = False
        for nr, nc in candidates:
            if progress_cb:
                progress_cb(nr, nc)
            try:
                s, rl = _try_solve(adapter, topo_graph, nr, nc, require_cell,
                                   min(15.0, time_limit))
                cur_r, cur_c, cur_sol, cur_rl = nr, nc, s, rl
                compressed = True
                break
            except adapter.SolveFailure:
                continue
        if not compressed:
            break

    return cur_r, cur_c, cur_sol, cur_rl


def _layout_nodes(nodes: list[dict], edges: list[dict], exclude_to: set) -> None:
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


def _build_ratio_graph(p: int, q: int, max_share: float | None = None) -> dict:
    a, b, N = find_ratio_scheme(p, q, max_share)
    t1, t2, t3 = N - q, p, q - p
    if min(t1, t2, t3) < 0 or p <= 0 or q <= 0:
        raise ValueError("需满足 0 < p < q 且 q 为整数")

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
    if bfs_min_splits is not None:
        try:
            plan = split_leaves(N, (t1, t2, t3), max_share, max(20, N))
        except Exception:
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
                f"无法通过细分使每份占比小于 {max_share:.4g}（N={N}）")
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

    _layout_nodes(nodes, edges, exclude_to={c1_id} if c1_id else set())

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
_SOLVERS: dict[str, dict] = {
    "milp": {
        "label": "MILP (Z3, exact)",
        "kind": "milp",
    },
    "rust-rank-smt": {
        "label": "Rust - Z3 (rank-smt)",
        "kind": "rust",
    },
    "rust-stable": {
        "label": "Rust - Karzanov (stable)",
        "kind": "rust",
    },
}


# -- API endpoints ----------------------------------------------------------
@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return {"max_frames": DEFAULT_CONFIG.sim_max_frames}


@app.get("/api/solvers")
async def api_solvers():
    return {"solvers": [
        {"id": sid, "label": info["label"], "kind": info["kind"]}
        for sid, info in _SOLVERS.items()
    ]}


@app.post("/api/solve")
async def api_solve(req: SolveRequest) -> Any:
    engine = req.engine or "milp"
    if engine != "milp":
        nodes = [n.model_dump() for n in req.nodes]
        edges = [
            {"id": e.id, "from": e.from_, "to": e.to} for e in req.edges
        ]
        native = await asyncio.to_thread(solve_native, engine, nodes, edges)
        return JSONResponse(native)

    t0 = time.perf_counter()
    node_ids = {n.id for n in req.nodes}

    text_lines: List[str] = []
    req_edges: List[EdgeModel] = []
    for edge in req.edges:
        if edge.from_ in node_ids and edge.to in node_ids:
            text_lines.append(f"{edge.from_} -> {edge.to}")
            req_edges.append(edge)
    text = "\n".join(text_lines)

    try:
        graph = Graph.from_text(text)
    except ValueError as e:
        return {
            "feasible": False,
            "error": str(e),
            "totalSolutions": 0,
            "solutions": [],
            "provedInfeasible": False,
        }

    logger.info("=" * 56)
    logger.info("Solve request - %d nodes, %d edges", len(req.nodes), len(req_edges))
    edge_desc = ", ".join(f"{e.from_}->{e.to}" for e in req_edges)
    logger.info("Edges: %s", edge_desc)

    solutions: List[Dict[str, Any]] = []
    patterns: List[List[bool]] = []
    proved_infeasible = False

    for i in range(MAX_SOLUTIONS):
        result = solve(graph, exclude_patterns=patterns if patterns else None)
        status_name = result.status

        if status_name == 'Infeasible':
            proved_infeasible = True
            break
        if status_name != 'Optimal':
            break

        payload = _build_solution_payload(req_edges, result)
        solutions.append(payload)

        free = _find_free_edge_indices(result)
        base = [e.is_blocked for e in result.edges]

        if free:
            combos = _generate_blocked_combos(base, free)
            patterns.extend(combos)
            logger.info("--- Solution %d --- (%d free edges: %s -> %d exclusion patterns)",
                        i + 1, len(free), sorted(free), len(combos))
        else:
            patterns.append(base)
            logger.info("--- Solution %d ---", i + 1)

        for ef in payload["edgeFlows"]:
            logger.info("  %s : flow=%s blocked=%s", ef["id"], ef["flow"]["text"], ef["isBlocked"])
        sink_flows = {nf["id"]: nf["flow"]["text"] for nf in payload["nodeFlows"]}
        logger.info("  Node flows: %s", sink_flows)

    elapsed = time.perf_counter() - t0

    before = len(solutions)
    solutions = _deduplicate_solutions(solutions)
    if before != len(solutions):
        logger.info("Deduplicated: %d -> %d solutions (%d duplicates removed)",
                    before, len(solutions), before - len(solutions))

    if not solutions:
        logger.warning("No feasible solution found (%.2fs)", elapsed)
        logger.info("=" * 56)
        return {
            "feasible": False,
            "error": "No feasible solution found",
            "totalSolutions": 0,
            "solutions": [],
            "provedInfeasible": False,
        }

    logger.info("Found %d solutions in %.2fs (proved=%s)", len(solutions), elapsed, proved_infeasible)
    logger.info("=" * 56)

    return {
        "feasible": True,
        "totalSolutions": len(solutions),
        "provedInfeasible": proved_infeasible,
        "solutions": solutions,
    }


@app.post("/api/simulate")
def api_simulate(req: SimulateRequest) -> Dict[str, Any]:
    node_ids = {n.id for n in req.nodes}
    text_lines: List[str] = []
    req_edges: List[EdgeModel] = []
    for edge in req.edges:
        if edge.from_ in node_ids and edge.to in node_ids:
            text_lines.append(f"{edge.from_} -> {edge.to}")
            req_edges.append(edge)
    text = "\n".join(text_lines)

    try:
        graph = Graph.from_text(text)
    except ValueError as exc:
        return {"error": str(exc)}

    result = simulate_frames(graph, max_frames=req.options.max_frames)

    edge_to_id: Dict[Tuple[str, str, int], str] = {}
    for i, e in enumerate(req_edges):
        edge_to_id[(e.from_, e.to, i)] = e.id

    node_ratios: Dict[str, Dict[str, Any]] = {}
    for node, (num, den) in result['cycle']['node_ratios'].items():
        g = math.gcd(num, den) if den > 0 else 1
        node_ratios[node] = {
            'numerator': num,
            'denominator': den,
            'text': f'{num}/{den}',
            'textReduced': f'{num // g}/{den // g}' if den > 0 else '0/0',
        }

    edge_ratios: Dict[str, Dict[str, Any]] = {}
    for edge, (num, den) in result['cycle']['edge_ratios'].items():
        key = (edge[0], edge[1], edge[2])
        edge_id = edge_to_id.get(key, f'{edge[0]}->{edge[1]}')
        g = math.gcd(num, den) if den > 0 else 1
        edge_ratios[edge_id] = {
            'numerator': num,
            'denominator': den,
            'text': f'{num}/{den}',
            'textReduced': f'{num // g}/{den // g}' if den > 0 else '0/0',
        }

    frames_json: List[Dict[str, Any]] = []
    for f in result['frames']:
        frame_nodes: Dict[str, Dict[str, Any]] = {}
        for nd in f['nodes']:
            frame_nodes[nd['id']] = {
                'hasItem': nd['has_item'],
                'rrIn': nd['rr_in_index'],
                'rrOut': nd['rr_out_index'],
            }
        frame_edges: Dict[str, Dict[str, Any]] = {}
        for ed in f['edges']:
            key = (ed['from'], ed['to'], ed['idx'])
            eid = edge_to_id.get(key, ed.get('id') or f"{ed['from']}->{ed['to']}_{ed['idx']}")
            frame_edges[eid] = {
                'queue': ed['queue'],
            }
        frames_json.append({
            'frame': f['frame'],
            'nodes': frame_nodes,
            'edges': frame_edges,
        })

    return {
        'cycleInfo': {
            'period': result['cycle']['period'],
            'cycleStartFrame': result['cycle']['cycle_start_frame'],
            'totalFrames': result['cycle']['total_frames'],
            'warmupFrames': result['cycle']['warmup_frames'],
            'nodeRatios': node_ratios,
            'edgeRatios': edge_ratios,
        },
        'frames': frames_json,
    }


@app.post("/api/ratio-split")
async def api_ratio_split(request: Request):
    body = await request.json()
    try:
        p = int(body.get("p"))
        q = int(body.get("q"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "请输入正整数 p、q"}, status_code=400)
    if p <= 0 or q <= 0 or p >= q:
        return JSONResponse({"error": "需满足 0 < p < q"}, status_code=400)
    try:
        graph = await asyncio.to_thread(_build_ratio_graph, p, q)
        return JSONResponse(graph)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _limit_node_type(name: str) -> str:
    if name == "In":
        return "In"
    if name == "Out":
        return "Out"
    if name.startswith("S"):
        return "S"
    if name.startswith("C"):
        return "C"
    raise ValueError(f"未知构造节点: {name}")


def _build_limit_module(p: int, q: int, optimize: bool, search_range: int,
                        reduction_depth: int, reduction_state_limit: int,
                        cross_check: bool) -> dict:
    from constructor import boundary_flow, construct_fraction, crosscheck_polynomial

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

    _layout_nodes(nodes, edges, exclude_to=set())

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
        check = crosscheck_polynomial(cert, result.target)
        info["crossCheck"] = {
            "backend": check.backend,
            "status": check.status,
            "flow": f"{check.flow.numerator}/{check.flow.denominator}",
            "iterations": check.iterations,
            "elapsed": check.elapsed_seconds,
        }

    return {"nodes": nodes, "edges": edges, "edgeFlows": edge_flows, "info": info}


@app.post("/api/limit-module")
async def api_limit_module(req: LimitModuleRequest):
    if not (0 < req.p < req.q):
        return JSONResponse({"error": "需满足 0 < p < q"}, status_code=400)
    try:
        graph = await asyncio.to_thread(
            _build_limit_module, req.p, req.q, req.optimize, req.search_range,
            req.reduction_depth, req.reduction_state_limit, req.cross_check,
        )
        return JSONResponse(graph)
    except Exception as e:
        logger.exception("limit-module failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/topoflow-layout")
async def api_topoflow_layout(request: Request):
    body = await request.json()
    nodes, edges = _raw_graph_from_body(body)
    require_cell = bool(body.get("requireBeltCell", True))
    time_limit = float(body.get("timeLimit", 30.0))
    min_size = int(body.get("minGrid", 3))

    try:
        adapter = _get_layout_adapter()
        topo_graph = _convert_for_topoflow(nodes, edges)
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)

    q: "queue.Queue" = queue.Queue()

    def run():
        def progress(rows: int, cols: int):
            q.put({"type": "progress", "rows": rows, "cols": cols})
        result = _find_min_grid(adapter, topo_graph, require_cell,
                                time_limit, min_size, progress)
        if result is None:
            q.put({"type": "result", "result": None})
        else:
            rows, cols, solution, route_length = result
            q.put({"type": "result", "result": {
                "status": "ok", "grid": [rows, cols],
                "routeLength": route_length, "solution": solution,
            }})

    async def gen():
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, run)
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                if fut.done():
                    item = q.get_nowait()
                else:
                    await asyncio.sleep(0.05)
                    continue
            yield json.dumps(item) + "\n"
            if item.get("type") == "result":
                break

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# -- Static files ------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True),
              name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting server: http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
