"""Universal construction driven by a bounded symbolic reduction search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

from ..leaves.unit import DEFAULT_EXHAUSTION_PAIR_LIMIT, UnitSolver
from ..model.certificate import CertificateCheck, FlowCertificate
from ..operators.composition import boundary_flow
from ..search.dp import ReductionPlan, ReductionPlanner
from .core import ConstructionCore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConstructionTiming:
    planning_seconds: float
    unit_search_seconds: float
    realization_seconds: float
    validation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class ConstructionResult:
    target: Fraction
    certificate: FlowCertificate
    reduction: ReductionPlan
    validation: CertificateCheck
    timing: ConstructionTiming


def construct_fraction(
    p: int,
    q: int,
    *,
    max_k: int | None = None,
    optimize: bool = False,
    search_range: int = 0,
    unit_solver: UnitSolver = "mixed",
    exhaustion_pair_limit: int = DEFAULT_EXHAUSTION_PAIR_LIMIT,
    reduction_depth: int = 3,
    reduction_state_limit: int = 4096,
) -> ConstructionResult:
    """Construct and certify an exact graph for ``0 < p/q < 1``."""
    total_started = perf_counter()
    if q <= 0 or p <= 0 or p >= q:
        raise ValueError("target must satisfy 0 < p < q")
    if max_k is not None and max_k < 1:
        raise ValueError("max_k must be positive")
    if search_range < 0:
        raise ValueError("search_range must be non-negative")
    if exhaustion_pair_limit < 1:
        raise ValueError("exhaustion_pair_limit must be positive")

    target = Fraction(p, q)
    core = ConstructionCore(max_k, unit_solver, exhaustion_pair_limit)
    planner = ReductionPlanner(
        core,
        optimize_interval=optimize,
        search_range=search_range,
        depth=reduction_depth,
        state_limit=reduction_state_limit,
    )
    planned = planner.plan(target)

    realization_started = perf_counter()
    certificate = planner.realize(planned.root)
    realization_seconds = perf_counter() - realization_started

    validation_started = perf_counter()
    check = certificate.check()
    if not check.valid:
        raise AssertionError(f"final construction is invalid: {check.errors}")
    actual_flow = boundary_flow(certificate)
    if actual_flow != target:
        raise AssertionError(
            f"final construction has flow {actual_flow}, expected {target}"
        )
    validation_seconds = perf_counter() - validation_started
    actual_cost = certificate.node_count, certificate.edge_count
    if actual_cost != planned.root.cost.score:
        raise AssertionError(
            f"planned cost {planned.root.cost.score} disagrees with graph {actual_cost}"
        )

    total_seconds = perf_counter() - total_started
    timing = ConstructionTiming(
        max(0.0, planned.stats.elapsed_seconds - planned.stats.unit_search_seconds),
        planned.stats.unit_search_seconds,
        realization_seconds,
        validation_seconds,
        total_seconds,
    )
    logger.info(
        "construction target=%s cost=%dn/%de planning=%.3fs unit=%.3fs "
        "realization=%.3fs validation=%.3fs total=%.3fs",
        target,
        actual_cost[0],
        actual_cost[1],
        timing.planning_seconds,
        timing.unit_search_seconds,
        timing.realization_seconds,
        timing.validation_seconds,
        timing.total_seconds,
    )
    return ConstructionResult(target, certificate, planned, check, timing)
