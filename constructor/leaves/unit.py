"""MILP-searched feedback-permutation modules with boundary flow ``1/n``."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from time import perf_counter
from typing import Any, Literal

from ..model.arithmetic import ceil_log3, factor_power_of_two, unit_topology_size
from ..model.certificate import EdgeState, FlowCertificate
from ..operators.composition import boundary_flow, replace_fixed_edge

Permutation = tuple[int, ...]
UnitSolver = Literal["mixed", "exhaustion", "milp"]
DEFAULT_EXHAUSTION_PAIR_LIMIT = 300_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecurrenceFlow:
    denominator: int
    converger_prefixes: tuple[int, ...]
    splitter_prefixes: tuple[int, ...]
    sweeps: int


@dataclass(frozen=True, slots=True)
class UnitSearchResult:
    denominator: int
    k: int
    pi: Permutation
    sigma: Permutation
    flow: RecurrenceFlow
    backend: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class UnitPlan:
    denominator: int
    twos: int
    odd: int
    witness: UnitSearchResult | None

    @property
    def topology_size(self) -> tuple[int, int]:
        odd_k = 0 if self.witness is None else self.witness.k
        return unit_topology_size(self.twos, odd_k)


def half_certificate() -> FlowCertificate:
    return FlowCertificate(
        (
            ("In", "C2_0"),
            ("S2_0", "C2_0"),
            ("C2_0", "S2_0"),
            ("S2_0", "Out"),
        ),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1), Fraction(1, 2)),
        (EdgeState.FULL, EdgeState.PARTIAL, EdgeState.FULL, EdgeState.EMPTY),
        frozenset({2}),
    )


def third_certificate() -> FlowCertificate:
    return certify_permutation((0,), (0,))


@cache
def plan_unit(
    denominator: int,
    *,
    max_k: int | None = None,
    solver: UnitSolver = "mixed",
    exhaustion_pair_limit: int = DEFAULT_EXHAUSTION_PAIR_LIMIT,
) -> UnitPlan:
    """Resolve a unit topology size and witness without constructing a graph."""
    if denominator < 2:
        raise ValueError("denominator must be at least 2")
    twos, odd = factor_power_of_two(denominator)
    witness = None
    if odd > 1:
        witness = search_odd_unit(
            odd,
            max_k=max_k,
            solver=solver,
            exhaustion_pair_limit=exhaustion_pair_limit,
        )
        if witness is None:
            raise RuntimeError(
                f"no 1/{odd} permutation witness found within max_k={max_k}"
            )
    return UnitPlan(denominator, twos, odd, witness)


def realize_unit(plan: UnitPlan) -> FlowCertificate:
    """Materialize a previously resolved unit plan."""
    factors: list[FlowCertificate] = [half_certificate() for _ in range(plan.twos)]
    if plan.witness is not None:
        factors.append(certify_permutation(plan.witness.pi, plan.witness.sigma))
    if not factors:
        raise ValueError("unit plan must contain at least one factor")
    result = factors[0]
    for factor in factors[1:]:
        result = replace_fixed_edge(result, factor)
    if boundary_flow(result) != Fraction(1, plan.denominator):
        raise AssertionError("unit factorization produced the wrong boundary flow")
    if (result.node_count, result.edge_count) != plan.topology_size:
        raise AssertionError("unit plan cost disagrees with constructed graph")
    return result


@cache
def construct_unit(
    denominator: int,
    *,
    max_k: int | None = None,
    solver: UnitSolver = "mixed",
    exhaustion_pair_limit: int = DEFAULT_EXHAUSTION_PAIR_LIMIT,
) -> FlowCertificate:
    """Construct ``1/denominator`` from a separately resolved plan."""
    return realize_unit(
        plan_unit(
            denominator,
            max_k=max_k,
            solver=solver,
            exhaustion_pair_limit=exhaustion_pair_limit,
        )
    )


def search_odd_unit(
    target: int,
    *,
    max_k: int | None = None,
    solver: UnitSolver = "mixed",
    exhaustion_pair_limit: int = DEFAULT_EXHAUSTION_PAIR_LIMIT,
) -> UnitSearchResult | None:
    if target <= 1 or target % 2 == 0:
        raise ValueError("odd unit target must be an odd integer greater than one")
    if solver not in ("mixed", "exhaustion", "milp"):
        raise ValueError(f"unknown unit solver: {solver!r}")
    if exhaustion_pair_limit < 1:
        raise ValueError("exhaustion_pair_limit must be positive")
    k = _minimum_k(target)
    if max_k is not None and max_k < k:
        raise ValueError(f"max_k must be at least {k} for denominator {target}")
    started = perf_counter()
    while max_k is None or k <= max_k:
        logger.info(
            "U(%d) search target=1/%d k=%d solver=%s canonical_pairs=%d",
            target,
            target,
            k,
            solver,
            candidate_pair_count(k),
        )
        if solver == "mixed":
            count = candidate_pair_count(k)
            pair = search_unit_at_k(
                target,
                k,
                backend="exhaustion",
                max_pairs=None
                if count <= exhaustion_pair_limit
                else exhaustion_pair_limit,
            )
            backend = (
                "exhaustion" if count <= exhaustion_pair_limit else "exhaustion-prefix"
            )
            if pair is None and count > exhaustion_pair_limit:
                pair = search_unit_at_k(target, k, backend="milp")
                backend = "milp"
        else:
            backend = solver
            pair = search_unit_at_k(target, k, backend=solver)
        if pair is not None:
            pi, sigma = pair
            flow = evaluate_permutation(pi, sigma)
            result = UnitSearchResult(
                target,
                k,
                pi,
                sigma,
                flow,
                backend,
                perf_counter() - started,
            )
            logger.info(
                "U(%d) witness target=1/%d k=%d backend=%s elapsed=%.3fs",
                target,
                target,
                k,
                backend,
                result.elapsed_seconds,
            )
            return result
        logger.info("U(%d) layer exhausted target=1/%d k=%d", target, target, k)
        k += 1
    return None


def candidate_pair_count(k: int) -> int:
    """Number of colour-swap-canonical permutation pairs at size ``k``."""
    if k < 1:
        raise ValueError("k must be positive")
    count = math.factorial(k)
    return count * (count + 1) // 2


def search_unit_at_k(
    target: int,
    k: int,
    *,
    backend: Literal["exhaustion", "milp"],
    max_pairs: int | None = None,
) -> tuple[Permutation, Permutation] | None:
    """Search exactly one layer with an explicitly selected backend."""
    if target <= 1 or target % 2 == 0:
        raise ValueError("odd unit target must be an odd integer greater than one")
    if k < 1:
        raise ValueError("k must be positive")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if backend == "exhaustion":
        return _search_at_k_exhaustive(target, k, max_pairs=max_pairs)
    if backend == "milp":
        if max_pairs is not None:
            raise ValueError("max_pairs applies only to exhaustion")
        return _solve_at_k_milp(target, k)
    raise ValueError(f"unknown fixed-k backend: {backend!r}")


def _search_at_k_exhaustive(
    target: int, k: int, *, max_pairs: int | None = None
) -> tuple[Permutation, Permutation] | None:
    """Enumerate canonical pairs, optionally stopping after a bounded prefix."""
    all_permutations = itertools.permutations(range(k))
    examined = 0
    for pi in all_permutations:
        for sigma in itertools.permutations(range(k)):
            if pi > sigma:
                continue
            examined += 1
            if evaluate_permutation(pi, sigma).denominator == target:
                return pi, sigma
            if max_pairs is not None and examined >= max_pairs:
                return None
    return None


def _solve_at_k_milp(target: int, k: int) -> tuple[Permutation, Permutation] | None:
    from ilpbridge import FEASIBLE, INFEASIBLE, OPTIMAL, Model, Solver

    model = Model()
    converger = [
        model.new_int_var(1, target, f"A_{j}") for j in range(k + 1)
    ]
    splitter = [
        model.new_int_var(1, target, f"B_{i}") for i in range(k + 1)
    ]
    selected = [
        [
            [
                model.new_bool_var(f"z_{colour}_{i}_{j}")
                for j in range(k)
            ]
            for i in range(k)
        ]
        for colour in range(2)
    ]
    feedback = [
        [
            [
                model.new_int_var(0, target, f"F_{colour}_{i}_{j}")
                for j in range(k)
            ]
            for i in range(k)
        ]
        for colour in range(2)
    ]
    a_le_b = [
        [model.new_bool_var(f"A_le_B_{i}_{j}") for j in range(k)]
        for i in range(k)
    ]
    model.add(converger[0] == 1)
    model.add(splitter[0] == 1)
    model.add(converger[k] == target)
    model.add(splitter[k] == target)
    for colour in range(2):
        for i in range(k):
            model.add(sum(selected[colour][i]) == 1)
        for j in range(k):
            model.add(sum(selected[colour][i][j] for i in range(k)) == 1)
    for i in range(k):
        for j in range(k):
            comparison = a_le_b[i][j]
            model.add(converger[j] - splitter[i] <= target * (1 - comparison))
            model.add(splitter[i] - converger[j] <= target * comparison)
            for colour in range(2):
                chosen = selected[colour][i][j]
                flow = feedback[colour][i][j]
                model.add(flow <= target * chosen)
                model.add(flow <= converger[j])
                model.add(flow <= splitter[i])
                model.add(flow >= converger[j] - target * (1 - chosen) - target * (
                    1 - comparison
                ))
                model.add(
                    flow >= splitter[i] - target * (1 - chosen) - target * comparison
                )
    for j in range(k):
        model.add(converger[j + 1] == converger[j] + sum(
            feedback[colour][i][j] for colour in range(2) for i in range(k)
        ))
    for i in range(k):
        model.add(splitter[i + 1] == splitter[i] + sum(
            feedback[colour][i][j] for colour in range(2) for j in range(k)
        ))
    model.add(sum(j * selected[0][0][j] for j in range(k)) <= sum(
        j * selected[1][0][j] for j in range(k)
    ))
    solver = Solver()
    solver.parameters.max_time_in_seconds = 300.0
    status = solver.solve(model)
    if status == INFEASIBLE:
        return None
    if status not in (OPTIMAL, FEASIBLE):
        raise RuntimeError(f"MILP stopped with status {status!r} at k={k}")

    def is_selected(variable: Any) -> bool:
        return solver.int_value(variable) > 0

    def selected_permutation(colour: int) -> Permutation:
        return tuple(
            next(j for j in range(k) if is_selected(selected[colour][i][j]))
            for i in range(k)
        )

    result = selected_permutation(0), selected_permutation(1)
    if evaluate_permutation(*result).denominator != target:
        raise RuntimeError("MILP witness failed exact recurrence verification")
    return result


def evaluate_permutation(pi: Sequence[int], sigma: Sequence[int]) -> RecurrenceFlow:
    if len(pi) != len(sigma) or not pi:
        raise ValueError("pi and sigma must have the same positive length")
    k = len(pi)
    expected = tuple(range(k))
    if tuple(sorted(pi)) != expected or tuple(sorted(sigma)) != expected:
        raise ValueError("pi and sigma must be permutations of range(k)")
    inverse_pi = [0] * k
    inverse_sigma = [0] * k
    for i, j in enumerate(pi):
        inverse_pi[j] = i
    for i, j in enumerate(sigma):
        inverse_sigma[j] = i
    converger = [1] * (k + 1)
    splitter = [1] * (k + 1)
    sweeps = 0
    while True:
        old_a, old_b = converger.copy(), splitter.copy()
        sweeps += 1
        for j in range(k):
            base = converger[j]
            converger[j + 1] = (
                base
                + min(base, splitter[inverse_pi[j]])
                + min(base, splitter[inverse_sigma[j]])
            )
        for i in range(k):
            base = splitter[i]
            splitter[i + 1] = (
                base + min(base, converger[pi[i]]) + min(base, converger[sigma[i]])
            )
        if converger == old_a and splitter == old_b:
            if converger[-1] != splitter[-1]:
                raise RuntimeError("stable boundary flows disagree")
            return RecurrenceFlow(
                converger[-1], tuple(converger), tuple(splitter), sweeps
            )


def certify_permutation(pi: Sequence[int], sigma: Sequence[int]) -> FlowCertificate:
    recurrence = evaluate_permutation(pi, sigma)
    q = recurrence.denominator
    a, b = recurrence.converger_prefixes, recurrence.splitter_prefixes
    k = len(pi)
    edges: list[tuple[str, str]] = [("In", "C3_0")]
    edges.extend((f"C3_{i}", f"C3_{i + 1}") for i in range(k - 1))
    edges.append((f"C3_{k - 1}", f"S3_{k - 1}"))
    edges.extend((f"S3_{i}", f"S3_{i - 1}") for i in range(k - 1, 0, -1))
    edges.append(("S3_0", "Out"))
    edges.extend((f"S3_{i}", f"C3_{pi[i]}") for i in range(k))
    edges.extend((f"S3_{i}", f"C3_{sigma[i]}") for i in range(k))
    integer_flows: list[int] = [
        a[0],
        *a[1:k],
        a[k],
        *(b[i] for i in range(k - 1, 0, -1)),
        b[0],
    ]
    integer_flows.extend(min(a[pi[i]], b[i]) for i in range(k))
    integer_flows.extend(min(a[sigma[i]], b[i]) for i in range(k))
    states: list[EdgeState] = [EdgeState.FULL] * k
    states.append(EdgeState.FULL)
    states.extend([EdgeState.EMPTY] * k)
    for permutation in (pi, sigma):
        for i, j in enumerate(permutation):
            if a[j] < b[i]:
                states.append(EdgeState.FULL)
            elif b[i] < a[j]:
                states.append(EdgeState.EMPTY)
            else:
                states.append(EdgeState.PARTIAL)
    certificate = FlowCertificate(
        tuple(edges),
        tuple(Fraction(value, q) for value in integer_flows),
        tuple(states),
        frozenset({k}),
    )
    errors = certificate.validation_errors()
    if errors:
        raise RuntimeError(f"invalid permutation certificate: {errors}")
    return certificate


def _minimum_k(target: int) -> int:
    return max(ceil_log3(target), 1)
