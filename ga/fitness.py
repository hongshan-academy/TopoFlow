from fractions import Fraction
from functools import lru_cache
from typing import Any, Tuple, Literal, Union

from graph import Graph
from solver import solve
from simulator import simulate
from config import DEFAULT_CONFIG as _cfg
from result import SolverResult, SimulatorResult

from ga.chromosome import decode


def _extract_source_flow(graph: Graph, result: Union[SolverResult, SimulatorResult]) -> float:
    source = next(iter(graph.sources))
    source_edge = graph.out_edges[source][0]
    for edge_result in result.edges:
        if edge_result.source == source and edge_result.target == source_edge[1]:
            return edge_result.flow
    return 0.0


def evaluate_cached(
    chromosome: Tuple[int, ...],
    target_pq: Tuple[int, int],
    threads: int = 1,
    max_denominator: int = 10000,
    mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP',
) -> Tuple[float, int]:
    graph = decode(chromosome)
    if mode == 'mixed':
        actual_mode: Literal['MILP', 'simulation'] = 'simulation' if len(graph.edges) <= _cfg.mixed_edge_threshold else 'MILP'
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


def _mp_eval_worker(args: Tuple[Tuple[int, ...], Tuple[int, int], int, int, Literal['MILP', 'simulation', 'mixed']]) -> Tuple[float, int]:
    chromosome, target_pq, threads, max_denominator, mode = args
    try:
        return evaluate_cached(chromosome, target_pq, threads=threads, max_denominator=max_denominator, mode=mode)
    except Exception:
        return (float("inf"), 0)


def make_evaluate(
    target_pq: Tuple[int, int],
    threads: int = 1,
    max_denominator: int = 10000,
    mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP',
) -> Any:
    @lru_cache(maxsize=_cfg.solver_cache_size)
    def _evaluate(chromosome: Tuple[int, ...]) -> Tuple[float, int]:
        return evaluate_cached(chromosome, target_pq, threads=threads, max_denominator=max_denominator, mode=mode)
    return _evaluate
