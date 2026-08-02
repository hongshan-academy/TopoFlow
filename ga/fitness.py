from fractions import Fraction
from functools import lru_cache
from typing import Any, Tuple, Literal, Union

from graph import Graph, Edge
from solver import solve
from simulator import simulate
from config import DEFAULT_CONFIG as _cfg
from result import SolverResult, SimulatorResult


def _extract_source_flow(graph: Graph, result: Union[SolverResult, SimulatorResult]) -> float:
    source = next(iter(graph.sources))
    source_edge = graph.out_edges[source][0]
    for edge_result in result.edges:
        if edge_result.source == source and edge_result.target == source_edge[1]:
            return edge_result.flow
    return 0.0


def evaluate_cached(
    edges_tuple: Tuple[Edge, ...],
    target_pq: Tuple[int, int],
    threads: int = 1,
    max_denominator: int = 10000,
    mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP',
) -> Tuple[float, int]:
    graph = Graph.from_edges(list(edges_tuple))
    if mode == 'mixed':
        actual_mode: Literal['MILP', 'simulation'] = (
            'simulation' if len(graph.edges) <= _cfg.mixed_edge_threshold else 'MILP'
        )
    else:
        actual_mode = mode
    if actual_mode == 'simulation':
        sim_result = simulate(graph, max_frames=_cfg.sim_max_frames)
        if not sim_result.converged:
            result: Union[SolverResult, SimulatorResult] = solve(graph, threads=threads)
        else:
            result = sim_result
    else:
        result = solve(graph, threads=threads)
    v = _extract_source_flow(graph, result)
    frac = Fraction(v).limit_denominator(max_denominator)
    p, q = target_pq
    target = Fraction(p, q)
    error = float(abs(target - frac))
    return (error, len(graph.nodes))

def make_evaluate(
    target_pq: Tuple[int, int],
    threads: int = 1,
    max_denominator: int = 10000,
    mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP',
) -> Any:
    @lru_cache(maxsize=_cfg.solver_cache_size)
    def _evaluate(edges_tuple: Tuple[Edge, ...]) -> Tuple[float, int]:
        return evaluate_cached(edges_tuple, target_pq, threads=threads, max_denominator=max_denominator, mode=mode)
    return _evaluate
