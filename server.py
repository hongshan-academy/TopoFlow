import logging
import math
import time
from fractions import Fraction
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from solver import solve
from graph import Graph
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


def approximate_fraction(value: float, max_denominator: int = 10000, tolerance: float = 1e-8):
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
    nodes: list[NodeModel]
    edges: list[EdgeModel]


MAX_SOLUTIONS = 50


def _find_free_edge_indices(result):
    free = set()
    for i, e in enumerate(result.edges):
        if e.flow >= 1.0 - 1e-8:
            free.add(i)
    return free


def _generate_blocked_combos(base, free_indices):
    free_list = sorted(free_indices)
    results = []
    for mask in range(1 << len(free_list)):
        pattern = list(base)
        for j, idx in enumerate(free_list):
            pattern[idx] = bool(mask & (1 << j))
        results.append(pattern)
    return results


def _build_solution_payload(req_edges, result):
    edge_flows = []
    for (req_edge, solver_edge) in zip(req_edges, result.edges):
        frac = approximate_fraction(solver_edge.flow)
        edge_flows.append({
            "id": req_edge.id,
            "flow": frac,
            "isBlocked": solver_edge.is_blocked,
        })

    node_flows_map = {}
    for e in result.edges:
        if e.target not in node_flows_map:
            node_flows_map[e.target] = 0.0
        node_flows_map[e.target] += e.flow

    node_flows_list = [
        {"id": nid, "flow": approximate_fraction(val)}
        for nid, val in node_flows_map.items()
    ]

    return {"edgeFlows": edge_flows, "nodeFlows": node_flows_list}


def _deduplicate_solutions(solutions):
    seen = {}
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


@app.post("/api/solve")
def api_solve(req: SolveRequest):
    t0 = time.perf_counter()
    node_ids = {n.node_id for n in req.nodes}

    text_lines = []
    req_edges = []
    for edge in req.edges:
        if edge.from_ in node_ids and edge.to in node_ids:
            text_lines.append(f"{edge.from_} -> {edge.to}")
            req_edges.append(edge)
    text = "\n".join(text_lines)

    graph = Graph.from_text(text)

    logger.info("=" * 56)
    logger.info("Solve request — %d nodes, %d edges", len(req.nodes), len(req_edges))
    edge_desc = ", ".join(f"{e.from_}->{e.to}" for e in req_edges)
    logger.info("Edges: %s", edge_desc)

    solutions = []
    patterns = []
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
