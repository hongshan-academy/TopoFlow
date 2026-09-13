"""Unified TopoFlow web backend (FastAPI).

Endpoints
  - GET  /api/config          discrete simulation config
  - GET  /api/solvers         list of available solvers
  - POST /api/solve           flow solving (engine = karzanov / rank-smt / milp)
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
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from flow_service import (
    build_limit_module,
    build_ratio_graph,
    deduplicate_solutions,
    enumerate_solutions,
)
from solver import solve_native
from simulator import simulate_frames
from graph import Graph
from config import DEFAULT_CONFIG

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("topoflow")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = RotatingFileHandler(LOG_DIR / "server.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

app = FastAPI(title="TopoFlow API")


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@app.exception_handler(RequestValidationError)
async def _on_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = first.get("msg", "Invalid request parameters")
    return JSONResponse(
        {"error": f"{location}: {message}" if location else message},
        status_code=422,
    )


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
    workers: int = 16


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


class RatioSplitRequest(BaseModel):
    p: int
    q: int


class TopoflowLayoutRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    nodes: List[dict] = Field(default_factory=list, validation_alias=AliasChoices("nodes", "rawNodes"))
    edges: List[dict] = Field(default_factory=list, validation_alias=AliasChoices("edges", "rawEdges"))
    require_belt_cell: bool = Field(default=True, validation_alias=AliasChoices("requireBeltCell", "require_belt_cell"))
    time_limit: float = Field(default=30.0, validation_alias=AliasChoices("timeLimit", "time_limit"))
    min_grid: int = Field(default=3, validation_alias=AliasChoices("minGrid", "min_grid"))
    workers: int = 16


# -- Common helpers ---------------------------------------------------------


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
                raise ValueError(f"Splitter {nid} has out-degree {out_d}, expected 2 or 3; cannot build physical layout")
            tf_type = "S3" if out_d == 3 else "S2"
        elif ntype == "C":
            if in_d < 2 or in_d > 3:
                raise ValueError(f"Converger {nid} has in-degree {in_d}, expected 2 or 3; cannot build physical layout")
            tf_type = "C3" if in_d == 3 else "C2"
        else:
            raise ValueError(f"Unsupported node type: {ntype}")
        id_mapping[nid] = f"{tf_type}_{type_counters[tf_type]}"
        type_counters[tf_type] += 1

    return {
        "nodes": [id_mapping[n["id"]] for n in nodes],
        "edges": [[id_mapping[e["from"]], id_mapping[e["to"]]] for e in edges],
    }


def _try_solve(adapter, topo_graph: dict, rows: int, cols: int,
               require_cell: bool, time_limit: float,
               workers: int = 16) -> tuple[dict, int]:
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
        workers=workers,
        search_mode="balanced",
    )
    return solution, route_length


def _find_min_grid(adapter, topo_graph: dict, require_cell: bool,
                   time_limit: float, min_size: int,
                   workers: int = 16,
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
                                           require_cell, time_limit, workers)
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
                                   min(15.0, time_limit), workers)
                cur_r, cur_c, cur_sol, cur_rl = nr, nc, s, rl
                compressed = True
                break
            except adapter.SolveFailure:
                continue
        if not compressed:
            break

    return cur_r, cur_c, cur_sol, cur_rl


_SOLVERS: dict[str, dict] = {
    "rust-karzanov": {
        "label": "Karzanov (polynomial) (Z3, exact)",
        "kind": "rust",
    },
    "rust-rank-smt": {
        "label": "rank-SMT (Z3, exact)",
        "kind": "rust",
    },
    "milp": {
        "label": "MILP (Z3, exact)",
        "kind": "milp",
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
        try:
            native = await asyncio.to_thread(solve_native, engine, nodes, edges, req.workers)
        except ValueError as e:
            return _error(str(e))
        except RuntimeError as e:
            return _error(str(e), 500)
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
        return _error(str(e))

    logger.info("=" * 56)
    logger.info("Solve request - %d nodes, %d edges", len(req.nodes), len(req_edges))
    edge_desc = ", ".join(f"{e.from_}->{e.to}" for e in req_edges)
    logger.info("Edges: %s", edge_desc)

    solutions, proved_infeasible = await asyncio.to_thread(
        enumerate_solutions, graph, req_edges, req.workers
    )
    elapsed = time.perf_counter() - t0

    before = len(solutions)
    solutions = deduplicate_solutions(solutions)
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
def api_simulate(req: SimulateRequest) -> Any:
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
        return _error(str(exc))

    max_frames = req.options.max_frames
    limit = DEFAULT_CONFIG.sim_max_frames
    if limit is not None and (max_frames is None or max_frames > limit):
        max_frames = limit
    result = simulate_frames(graph, max_frames=max_frames)

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
async def api_ratio_split(req: RatioSplitRequest):
    if not (0 < req.p < req.q):
        return _error("p and q must satisfy 0 < p < q")
    try:
        graph = await asyncio.to_thread(build_ratio_graph, req.p, req.q)
        return JSONResponse(graph)
    except Exception as e:
        return _error(str(e), 500)


@app.post("/api/limit-module")
async def api_limit_module(req: LimitModuleRequest):
    if not (0 < req.p < req.q):
        return _error("p and q must satisfy 0 < p < q")
    try:
        graph = await asyncio.to_thread(
            build_limit_module, req.p, req.q, req.optimize, req.search_range,
            req.reduction_depth, req.reduction_state_limit, req.cross_check,
        )
        return JSONResponse(graph)
    except Exception as e:
        logger.exception("limit-module failed")
        return _error(str(e), 500)


@app.post("/api/topoflow-layout")
async def api_topoflow_layout(req: TopoflowLayoutRequest):
    try:
        adapter = _get_layout_adapter()
        topo_graph = _convert_for_topoflow(req.nodes, req.edges)
    except Exception as e:
        return _error(str(e))

    q: "queue.Queue" = queue.Queue()

    def run():
        def progress(rows: int, cols: int):
            q.put({"type": "progress", "rows": rows, "cols": cols})
        result = _find_min_grid(adapter, topo_graph, req.require_belt_cell,
                                req.time_limit, req.min_grid, req.workers, progress)
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
    print("Starting server: http://localhost:8081")
    uvicorn.run("server:app", host="0.0.0.0", port=8081, reload=False)
