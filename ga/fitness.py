from fractions import Fraction
from functools import lru_cache
from typing import Tuple

from graph import Graph
from solver import solve, SolverResult
from config import DEFAULT_CONFIG as _cfg

from ga.utils import tuple_to_graph


def _extract_source_flow(graph: Graph, result: SolverResult) -> float:
    source = next(iter(graph.sources))
    source_edge = graph.out_edges[source][0]
    for edge_result in result.edges:
        if edge_result.source == source and edge_result.target == source_edge[1]:
            return edge_result.flow
    return 0.0


def evaluate_cached(
    edges_tuple: Tuple[Tuple[str, str], ...],
    target_pq: Tuple[int, int],
    threads: int = 1,
    max_denominator: int = 10000,
) -> Tuple[float, int]:
    graph = tuple_to_graph(edges_tuple)
    result = solve(graph, threads=threads)
    v = _extract_source_flow(graph, result)
    frac = Fraction(v).limit_denominator(max_denominator)
    p, q = target_pq
    error = abs(p * frac.denominator - q * frac.numerator)
    return (float(error), len(graph.nodes))


def _mp_eval_worker(args: Tuple[Tuple[Tuple[str, str], ...], Tuple[int, int], int, int]) -> Tuple[float, int]:
    edges_tuple, target_pq, threads, max_denominator = args
    try:
        return evaluate_cached(edges_tuple, target_pq, threads=threads, max_denominator=max_denominator)
    except Exception:
        return (float("inf"), 0)


def make_evaluate(target_pq: Tuple[int, int], threads: int = 1, max_denominator: int = 10000):
    @lru_cache(maxsize=_cfg.solver_cache_size)
    def _evaluate(edges_tuple: Tuple[Tuple[str, str], ...]) -> Tuple[float, int]:
        return evaluate_cached(edges_tuple, target_pq, threads=threads, max_denominator=max_denominator)
    return _evaluate
