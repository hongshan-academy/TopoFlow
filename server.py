import logging
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from solver import solve
from simulator import simulate_frames
from graph import Graph
from config import DEFAULT_CONFIG
from result import SolverResult
import pulp

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


def approximate_fraction(value: float, max_denominator: int = 10000, tolerance: float = 1e-8) -> Dict[str, Any]:
    if not math.isfinite(value):
        return {"numerator": 0, "denominator": 1, "text": "0/1"}
    if abs(value) < tolerance:
        return {"numerator": 0, "denominator": 1, "text": "0/1"}
    if abs(value - 1) < tolerance:
        return {"numerator": 1, "denominator": 1, "text": "1/1"}
    frac = Fraction(value).limit_denominator(max_denominator)
    return {
        "numerator": frac.numerator,
        "denominator": frac.denominator,
        "text": f"{frac.numerator}/{frac.denominator}",
    }


class NodeModel(BaseModel):
    node_id: str
    node_type: str
    x: float = 0
    y: float = 0


class EdgeModel(BaseModel):
    id: str
    from_: str = Field(alias="from")
    to: str


class SolveRequest(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]


class SimulateOptions(BaseModel):
    max_frames: Optional[int] = None


class SimulateRequest(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    options: SimulateOptions = Field(default_factory=SimulateOptions)


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


def _build_solution_payload(
    req_edges: List[EdgeModel], result: SolverResult
) -> Dict[str, Any]:
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


app = FastAPI()


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return {"max_frames": DEFAULT_CONFIG.sim_max_frames}


@app.post("/api/solve")
def api_solve(req: SolveRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    node_ids = {n.node_id for n in req.nodes}

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
    logger.info("Solve request — %d nodes, %d edges", len(req.nodes), len(req_edges))
    edge_desc = ", ".join(f"{e.from_}->{e.to}" for e in req_edges)
    logger.info("Edges: %s", edge_desc)

    solutions: List[Dict[str, Any]] = []
    patterns: List[List[bool]] = []
    proved_infeasible = False

    for i in range(MAX_SOLUTIONS):
        result = solve(graph, exclude_patterns=patterns if patterns else None)
        status_name = pulp.LpStatus[result.status]

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
            logger.info("--- Solution %d --- (%d free edges: %s → %d exclusion patterns)",
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
    node_ids = {n.node_id for n in req.nodes}
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
        return {"error": str(e)}

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
            # ensure eid is a str (edge_to_id.get or ed['id'] may be None)
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
