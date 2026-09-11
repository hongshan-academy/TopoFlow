"""Shared certified construction primitives used by every planning strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..leaves.unit import (
    DEFAULT_EXHAUSTION_PAIR_LIMIT,
    UnitPlan,
    UnitSolver,
    plan_unit,
    realize_unit,
)
from ..model.certificate import FlowCertificate
from ..operators.composition import (
    full_certificate,
    parallel_sum,
    replace_fixed_edge,
    restricted_sub,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConstructionCore:
    """Cached construction facade used by the fraction planner."""

    max_k: int | None = None
    unit_solver: UnitSolver = "mixed"
    exhaustion_pair_limit: int = DEFAULT_EXHAUSTION_PAIR_LIMIT
    _units: dict[int, FlowCertificate] = field(default_factory=dict, init=False)
    _unit_plans: dict[int, UnitPlan] = field(default_factory=dict, init=False)

    def unit_plan(self, denominator: int) -> UnitPlan:
        if denominator not in self._unit_plans:
            self._unit_plans[denominator] = plan_unit(
                denominator,
                max_k=self.max_k,
                solver=self.unit_solver,
                exhaustion_pair_limit=self.exhaustion_pair_limit,
            )
        return self._unit_plans[denominator]

    def unit(self, denominator: int) -> FlowCertificate:
        if denominator not in self._units:
            logger.info("core U(%d) target=1/%d", denominator, denominator)
            certificate = realize_unit(self.unit_plan(denominator))
            self._units[denominator] = certificate
        return self._units[denominator]

    def add(self, left: FlowCertificate, right: FlowCertificate) -> FlowCertificate:
        return parallel_sum(left, right)

    def multiply(
        self, outer: FlowCertificate, inner: FlowCertificate
    ) -> FlowCertificate:
        return replace_fixed_edge(outer, inner)

    def full(self) -> FlowCertificate:
        return full_certificate()

    def sub(
        self, minuend: FlowCertificate, subtrahend: FlowCertificate
    ) -> FlowCertificate:
        return restricted_sub(minuend, subtrahend)
