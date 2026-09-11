"""Immutable DP construction recipes and their topology costs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, Protocol

from .certificate import FlowCertificate


class IntervalPlanLike(Protocol):
    @property
    def target(self) -> Fraction: ...

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    @property
    def strategy(self) -> str: ...


RecipeKind = Literal["full", "unit", "interval", "add", "scale", "sub"]


@dataclass(frozen=True, slots=True)
class TopologyCost:
    nodes: int
    edges: int
    fixed_edges: int
    exact: bool = True

    @property
    def score(self) -> tuple[int, int]:
        return self.nodes, self.edges

    @classmethod
    def from_certificate(cls, certificate: FlowCertificate) -> TopologyCost:
        return cls(
            certificate.node_count,
            certificate.edge_count,
            len(certificate.fixed),
        )


@dataclass(frozen=True, slots=True)
class ReductionRecipe:
    kind: RecipeKind
    target: Fraction
    cost: TopologyCost
    children: tuple[ReductionRecipe, ...] = ()
    interval: IntervalPlanLike | None = None
    label: str = ""

    def __post_init__(self) -> None:
        valid = False
        if self.kind == "full":
            valid = not self.children and self.interval is None and self.target == 1
        elif self.kind == "unit":
            valid = (
                not self.children
                and self.interval is None
                and self.target.numerator == 1
            )
        elif self.kind == "interval":
            valid = (
                not self.children
                and self.interval is not None
                and self.target == self.interval.target
            )
        elif self.kind == "add":
            valid = (
                self._valid_binary_denominators()
                and self.target == self.children[0].target + self.children[1].target
                and self.target < 1
            )
        elif self.kind == "scale":
            valid = (
                len(self.children) == 2
                and self.interval is None
                and self.children[0].kind == "unit"
                and self.target
                == self.children[1].target / self.children[0].target.denominator
            )
        elif self.kind == "sub":
            valid = (
                self._valid_binary_denominators()
                and self.children[0].target > 2 * self.children[1].target
                and self.target == self.children[0].target - self.children[1].target
            )
        if not valid:
            raise ValueError(f"invalid {self.kind} recipe for target {self.target}")
        expected = self._derived_cost(self.kind, self.children, interval=self.interval)
        if expected is not None and self.cost != expected:
            raise ValueError(
                f"{self.kind} recipe cost {self.cost} does not match {expected}"
            )

    def _valid_binary_denominators(self) -> bool:
        return (
            len(self.children) == 2
            and self.interval is None
            and max(child.target.denominator for child in self.children)
            <= self.target.denominator
        )

    @staticmethod
    def _derived_cost(
        kind: RecipeKind,
        children: tuple[ReductionRecipe, ...],
        *,
        interval: IntervalPlanLike | None = None,
    ) -> TopologyCost | None:
        if kind in ("full", "unit"):
            return None
        if kind == "interval":
            assert interval is not None
            return TopologyCost(interval.node_count, interval.edge_count, 1)
        if kind in ("add", "sub"):
            left, right = children
            return TopologyCost(
                left.cost.nodes + right.cost.nodes,
                left.cost.edges + right.cost.edges + 2,
                left.cost.fixed_edges + right.cost.fixed_edges,
                left.cost.exact and right.cost.exact,
            )
        if kind == "scale":
            unit, inner = children
            return TopologyCost(
                unit.cost.nodes + inner.cost.nodes - 2,
                unit.cost.edges + inner.cost.edges - 1,
                inner.cost.fixed_edges,
                unit.cost.exact and inner.cost.exact,
            )
        raise ValueError(f"unknown recipe kind: {kind}")

    @classmethod
    def for_full(cls) -> ReductionRecipe:
        return cls("full", Fraction(1), TopologyCost(2, 1, 1), label="full")

    @classmethod
    def for_unit(
        cls, denominator: int, cost: TopologyCost, *, label: str = "unit"
    ) -> ReductionRecipe:
        if denominator < 2:
            raise ValueError("unit denominator must be at least 2")
        return cls("unit", Fraction(1, denominator), cost, label=label)

    @classmethod
    def for_interval(cls, plan: IntervalPlanLike) -> ReductionRecipe:
        cost = cls._derived_cost("interval", (), interval=plan)
        assert cost is not None
        return cls(
            "interval",
            plan.target,
            cost,
            interval=plan,
            label=plan.strategy,
        )

    @classmethod
    def for_add(
        cls, left: ReductionRecipe, right: ReductionRecipe, *, label: str
    ) -> ReductionRecipe:
        target = left.target + right.target
        cost = cls._derived_cost("add", (left, right))
        assert cost is not None
        return cls("add", target, cost, (left, right), label=label)

    @classmethod
    def for_scale(
        cls,
        unit: ReductionRecipe,
        inner: ReductionRecipe,
        *,
        label: str,
    ) -> ReductionRecipe:
        if unit.kind != "unit":
            raise ValueError("scale recipe requires a unit recipe")
        cost = cls._derived_cost("scale", (unit, inner))
        assert cost is not None
        return cls(
            "scale",
            inner.target / unit.target.denominator,
            cost,
            (unit, inner),
            label=label,
        )

    @classmethod
    def for_sub(
        cls, minuend: ReductionRecipe, subtrahend: ReductionRecipe, *, label: str
    ) -> ReductionRecipe:
        cost = cls._derived_cost("sub", (minuend, subtrahend))
        assert cost is not None
        return cls(
            "sub",
            minuend.target - subtrahend.target,
            cost,
            (minuend, subtrahend),
            label=label,
        )

    @property
    def factor(self) -> int | None:
        return self.children[0].target.denominator if self.kind == "scale" else None

    def with_label(self, label: str) -> ReductionRecipe:
        return replace(self, label=label)

    def __str__(self) -> str:
        if self.kind == "full":
            return "1"
        if self.kind == "unit":
            return f"U({self.target.denominator})"
        if self.kind == "interval":
            assert self.interval is not None
            return (
                f"I({self.target.numerator}, {self.target.denominator})"
                f"[{self.interval.strategy}]"
            )
        if self.kind == "add":
            return f"({self.children[0]}) + ({self.children[1]})"
        if self.kind == "scale":
            return f"({self.children[1]})/{self.factor}"
        if self.kind == "sub":
            return f"({self.children[0]}) - ({self.children[1]})"
        raise AssertionError(f"unknown recipe kind: {self.kind}")

    def steps(self) -> tuple[str, ...]:
        steps = [step for child in self.children for step in child.steps()]
        steps.append(str(self))
        return tuple(steps)
