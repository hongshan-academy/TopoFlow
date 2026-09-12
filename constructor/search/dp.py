"""Bounded symbolic search for small certified rational-flow graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from time import perf_counter
from typing import cast

from ..leaves.interval import IntervalPlan, plan_interval, realize_interval
from ..model.arithmetic import ceil_log3, factor_power_of_two, unit_topology_size
from ..model.certificate import FlowCertificate
from ..model.recipe import ReductionRecipe, TopologyCost
from ..service.core import ConstructionCore

logger = logging.getLogger(__name__)
SMALL_FACTOR_LIMIT = 12
SIMPLE_PRIME_DENOMINATORS = (2, 3, 5, 7, 11, 13)
ADDITION_CANDIDATE_LIMIT = 6
SUB_CANDIDATE_LIMIT = 6
SIMPLE_FRACTIONS = tuple(
    sorted(
        Fraction(numerator, denominator)
        for denominator in SIMPLE_PRIME_DENOMINATORS
        for numerator in range(1, denominator)
    )
)


@dataclass(frozen=True, slots=True)
class PlannerStats:
    elapsed_seconds: float
    unit_search_seconds: float
    expanded_states: int
    candidate_count: int
    pruned_candidates: int
    unit_resolutions: int
    iterations: int


@dataclass(frozen=True, slots=True)
class ReductionPlan:
    target: Fraction
    strategy: str
    root: ReductionRecipe
    stats: PlannerStats

    @property
    def cost(self) -> TopologyCost:
        return self.root.cost

    def steps(self) -> tuple[str, ...]:
        return self.root.steps()


@dataclass(slots=True)
class ReductionPlanner:
    core: ConstructionCore
    optimize_interval: bool = False
    search_range: int = 0
    depth: int = 3
    state_limit: int = 4096
    _unit_costs: dict[int, TopologyCost] = field(default_factory=dict, init=False)
    _unavailable_units: dict[int, Exception] = field(default_factory=dict, init=False)
    _intervals: dict[Fraction, IntervalPlan] = field(default_factory=dict, init=False)
    _memo: dict[tuple[Fraction, int], ReductionRecipe] = field(
        default_factory=dict, init=False
    )
    _seen_states: set[tuple[Fraction, int]] = field(default_factory=set, init=False)
    _expanded_states: int = field(default=0, init=False)
    _candidate_count: int = field(default=0, init=False)
    _pruned_candidates: int = field(default=0, init=False)
    _unit_resolutions: int = field(default=0, init=False)
    _unit_seconds: float = field(default=0.0, init=False)
    _iterations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("reduction_depth must be non-negative")
        if self.state_limit < 1:
            raise ValueError("reduction_state_limit must be positive")

    def plan(self, target: Fraction) -> ReductionPlan:
        started = perf_counter()
        root: ReductionRecipe | None = None
        while True:
            self._iterations += 1
            self._memo.clear()
            self._seen_states.clear()
            root = self._solve(target, self.depth)
            unknown = sorted(self._unknown_units(root))
            logger.info(
                "planner iteration=%d target=%s recipe=%s optimistic=%dn/%de "
                "unresolved_units=%s",
                self._iterations,
                target,
                root,
                root.cost.nodes,
                root.cost.edges,
                unknown,
            )
            if not unknown:
                break
            for denominator in unknown:
                unit_started = perf_counter()
                try:
                    unit_plan = self.core.unit_plan(denominator)
                except (RuntimeError, ValueError) as exc:
                    self._unavailable_units[denominator] = exc
                    raise
                else:
                    nodes, edges = unit_plan.topology_size
                    self._unit_costs[denominator] = TopologyCost(nodes, edges, 1)
                finally:
                    self._unit_seconds += perf_counter() - unit_started
                    self._unit_resolutions += 1

        elapsed = perf_counter() - started
        stats = PlannerStats(
            elapsed,
            self._unit_seconds,
            self._expanded_states,
            self._candidate_count,
            self._pruned_candidates,
            self._unit_resolutions,
            self._iterations,
        )
        logger.info(
            "planner selected target=%s recipe=%s strategy=%s cost=%dn/%de "
            "states=%d candidates=%d elapsed=%.3fs",
            target,
            root,
            root.label,
            root.cost.nodes,
            root.cost.edges,
            stats.expanded_states,
            stats.candidate_count,
            elapsed,
        )
        return ReductionPlan(target, root.label, root, stats)

    def realize(self, recipe: ReductionRecipe) -> FlowCertificate:
        if recipe.kind == "full":
            return self.core.full()
        if recipe.kind == "unit":
            return self.core.unit(recipe.target.denominator)
        if recipe.kind == "interval":
            assert recipe.interval is not None
            return realize_interval(cast(IntervalPlan, recipe.interval))
        if recipe.kind == "add":
            return self.core.add(
                self.realize(recipe.children[0]), self.realize(recipe.children[1])
            )
        if recipe.kind == "scale":
            unit, inner = recipe.children
            return self.core.multiply(self.realize(unit), self.realize(inner))
        if recipe.kind == "sub":
            return self.core.sub(
                self.realize(recipe.children[0]), self.realize(recipe.children[1])
            )
        raise AssertionError(f"unknown recipe kind: {recipe.kind}")

    def _solve(self, value: Fraction, depth: int) -> ReductionRecipe:
        key = value, depth
        if key in self._memo:
            return self._memo[key]
        if value.numerator == 1:
            result = self._require_unit(value.denominator)
            self._memo[key] = result
            return result
        remainder = 1 - value
        if value > Fraction(1, 2) and remainder.numerator == 1:
            result = self._sub(
                ReductionRecipe.for_full(),
                self._require_unit(remainder.denominator),
                "sub-full-unit",
            )
            self._memo[key] = result
            return result
        if len(self._seen_states) >= self.state_limit:
            raise RuntimeError(
                f"no construction for {value} within reduction state limit "
                f"{self.state_limit}"
            )
        self._seen_states.add(key)
        self._expanded_states += 1

        candidates: list[ReductionRecipe] = [self._base_recipe(value)]
        binary = self._binary_remainder(value)
        if binary is not None:
            candidates.append(binary)
        if depth:
            if value > Fraction(1, 2):
                child = self._solve(1 - value, depth - 1)
                candidates.append(
                    self._sub(ReductionRecipe.for_full(), child, "sub-full")
                )

            for factor in self._multiplier_candidates(value):
                scaled = factor * value
                child = self._solve(scaled, depth - 1)
                candidates.append(self._scale(factor, child, "scale"))

            for left, right in self._addition_splits(value):
                candidates.append(
                    self._add(
                        self._solve(left, depth - 1),
                        self._solve(right, depth - 1),
                        f"add-split-{left}-{right}",
                    )
                )

            for minuend, subtrahend in self._sub_splits(value):
                candidates.append(
                    self._sub(
                        self._solve(minuend, depth - 1),
                        self._solve(subtrahend, depth - 1),
                        f"sub-split-{minuend}-{subtrahend}",
                    )
                )

        result = self._choose(value, candidates)
        self._memo[key] = result
        return result

    def _base_recipe(self, value: Fraction) -> ReductionRecipe:
        if value > Fraction(1, 2):
            return self._sub(
                ReductionRecipe.for_full(),
                self._base_recipe(1 - value),
                "base-sub-full",
            )
        if value < Fraction(1, 3):
            factor = _ceil_ratio(value.denominator, 3 * value.numerator)
            scaled = factor * value
            inner = (
                self._sub(
                    ReductionRecipe.for_full(),
                    self._interval_recipe(1 - scaled),
                    "base-sub-full",
                )
                if scaled > Fraction(1, 2)
                else self._interval_recipe(scaled)
            )
            return self._scale(factor, inner, "base-scale")
        return self._interval_recipe(value)

    def _binary_remainder(self, value: Fraction) -> ReductionRecipe | None:
        p, q = value.numerator, value.denominator
        exponent = (q - 1).bit_length()
        power = 1 << exponent
        quotient, remainder = divmod(p * power, q)
        if remainder == 0:
            quotient_part = self._dyadic_numerator(quotient, exponent)
            return quotient_part.with_label("binary-remainder")
        quotient_value = Fraction(quotient, power)
        remainder_value = value - quotient_value
        if not self._addition_allowed(
            quotient_value, remainder_value, "binary-remainder"
        ):
            return None
        quotient_part = self._dyadic_numerator(quotient, exponent)
        twos, odd = factor_power_of_two(q)
        remainder_part = self._dyadic_numerator(remainder, twos + exponent)
        if odd > 1:
            remainder_part = self._scale(odd, remainder_part, "binary-odd-core")
        return self._add(quotient_part, remainder_part, "binary-remainder")

    def _dyadic_numerator(self, numerator: int, exponent: int) -> ReductionRecipe:
        result: ReductionRecipe | None = None
        for bit in range(exponent):
            if result is not None:
                result = self._scale(2, result, "dyadic-scale")
            if numerator & (1 << bit):
                half = self._require_unit(2)
                result = (
                    half if result is None else self._add(result, half, "dyadic-add")
                )
        if result is None:
            raise AssertionError("positive dyadic numerator produced no recipe")
        return result

    def _interval_recipe(self, value: Fraction) -> ReductionRecipe:
        if value not in self._intervals:
            self._intervals[value] = plan_interval(
                value, optimize=self.optimize_interval, search_range=self.search_range
            )
        plan = self._intervals[value]
        return ReductionRecipe.for_interval(plan)

    def _require_unit(self, denominator: int) -> ReductionRecipe:
        if denominator in self._unavailable_units:
            raise self._unavailable_units[denominator]
        cost = self._unit_costs.get(denominator)
        if cost is None:
            cost = _unit_lower_cost(denominator)
        return ReductionRecipe.for_unit(denominator, cost)

    def _choose(
        self, value: Fraction, candidates: list[ReductionRecipe]
    ) -> ReductionRecipe:
        candidates = [
            candidate for candidate in candidates if candidate.target == value
        ]
        if not candidates:
            raise RuntimeError(f"no available construction for {value}")
        self._candidate_count += len(candidates)
        self._pruned_candidates += len(candidates) - 1
        result = min(
            candidates,
            key=lambda item: (
                item.cost.nodes,
                item.cost.edges,
                not item.cost.exact,
                item.label,
            ),
        )
        logger.debug(
            "planner state=%s candidates=%s selected=%s",
            value,
            tuple(
                (
                    str(item),
                    item.label,
                    item.cost.nodes,
                    item.cost.edges,
                    item.cost.exact,
                )
                for item in candidates
            ),
            result,
        )
        return result

    @staticmethod
    def _multiplier_candidates(value: Fraction) -> tuple[int, ...]:
        p, q = value.numerator, value.denominator
        values = set(range(2, SMALL_FACTOR_LIMIT + 1))
        critical = (
            _ceil_ratio(q, 3 * p),
            _ceil_ratio(q, 2 * p),
            (2 * q) // (3 * p),
            (q - 1) // p,
        )
        for center in critical:
            values.update((center - 1, center, center + 1))
        return tuple(
            sorted(factor for factor in values if factor >= 2 and factor * value < 1)
        )

    def _addition_splits(
        self, value: Fraction
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        pairs = set(self._factor_addition_splits(value))
        for atom in SIMPLE_FRACTIONS:
            if atom >= value:
                continue
            residual = value - atom
            if self._addition_allowed(atom, residual, "small-fraction"):
                pairs.add(_ordered_pair(atom, residual))
        return tuple(sorted(pairs, key=_addition_pair_key)[:ADDITION_CANDIDATE_LIMIT])

    @staticmethod
    def _factor_addition_splits(
        value: Fraction,
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        p, q = value.numerator, value.denominator
        blocks = _prime_power_blocks(q)
        if len(blocks) < 2:
            return ()
        pairs: set[tuple[Fraction, Fraction]] = set()
        # Keep the first prime-power block on the left so complementary
        # partitions are visited exactly once.
        for mask in range(1 << (len(blocks) - 1)):
            left_denominator = blocks[0]
            for index, block in enumerate(blocks[1:]):
                if mask & (1 << index):
                    left_denominator *= block
            if left_denominator == q:
                continue
            right_denominator = q // left_denominator
            left_numerator = (
                p * pow(right_denominator, -1, left_denominator) % left_denominator
            )
            if left_numerator == 0:
                continue
            remainder = p - right_denominator * left_numerator
            if remainder <= 0 or remainder % left_denominator:
                continue
            right_numerator = remainder // left_denominator
            if not 0 < right_numerator < right_denominator:
                continue
            left = Fraction(left_numerator, left_denominator)
            right = Fraction(right_numerator, right_denominator)
            pairs.add(_ordered_pair(left, right))
        return tuple(sorted(pairs, key=_addition_pair_key))

    def _sub_splits(
        self, value: Fraction
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        pairs = set(self._factor_sub_splits(value))
        for atom in SIMPLE_FRACTIONS:
            if atom < value and value + atom < 1:
                minuend, subtrahend = value + atom, atom
                if self._sub_allowed(minuend, subtrahend, "small-fraction"):
                    pairs.add((minuend, subtrahend))
            if value < atom < 2 * value:
                minuend, subtrahend = atom, atom - value
                if self._sub_allowed(minuend, subtrahend, "small-fraction"):
                    pairs.add((minuend, subtrahend))
        return tuple(
            sorted(pairs, key=_addition_pair_key)[:SUB_CANDIDATE_LIMIT]
        )

    @staticmethod
    def _factor_sub_splits(
        value: Fraction,
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        p, q = value.numerator, value.denominator
        blocks = _prime_power_blocks(q)
        if len(blocks) < 2:
            return ()
        pairs: set[tuple[Fraction, Fraction]] = set()
        for mask in range(1, (1 << len(blocks)) - 1):
            left_denominator = 1
            for index, block in enumerate(blocks):
                if mask & (1 << index):
                    left_denominator *= block
            right_denominator = q // left_denominator
            left_numerator = (
                p * pow(right_denominator, -1, left_denominator) % left_denominator
            )
            if left_numerator == 0:
                continue
            difference = right_denominator * left_numerator - p
            if difference <= 0 or difference % left_denominator:
                continue
            right_numerator = difference // left_denominator
            if not 0 < right_numerator < right_denominator:
                continue
            minuend = Fraction(left_numerator, left_denominator)
            subtrahend = Fraction(right_numerator, right_denominator)
            if ReductionPlanner._sub_allowed(
                minuend, subtrahend, "factor-split"
            ):
                pairs.add((minuend, subtrahend))
        return tuple(sorted(pairs, key=_addition_pair_key))

    @staticmethod
    def _sub_allowed(
        minuend: Fraction, subtrahend: Fraction, label: str
    ) -> bool:
        target = minuend - subtrahend
        allowed = (
            0 < target < 1
            and minuend > 2 * subtrahend
            and max(minuend.denominator, subtrahend.denominator) <= target.denominator
        )
        if not allowed:
            logger.debug(
                "planner skipped sub label=%s target=%s operands=(%s, %s) "
                "denominators=(%d, %d)",
                label,
                target,
                minuend,
                subtrahend,
                minuend.denominator,
                subtrahend.denominator,
            )
        return allowed

    @staticmethod
    def _addition_allowed(left: Fraction, right: Fraction, label: str) -> bool:
        target = left + right
        allowed = (
            target < 1
            and max(left.denominator, right.denominator) <= target.denominator
        )
        if not allowed:
            logger.debug(
                "planner skipped add label=%s target=%s operands=(%s, %s) "
                "denominators=(%d, %d)",
                label,
                target,
                left,
                right,
                left.denominator,
                right.denominator,
            )
        return allowed

    def _unknown_units(self, recipe: ReductionRecipe) -> set[int]:
        result: set[int] = set()
        if recipe.kind == "unit" and not recipe.cost.exact:
            result.add(recipe.target.denominator)
        for child in recipe.children:
            result.update(self._unknown_units(child))
        return result

    @staticmethod
    def _add(
        left: ReductionRecipe, right: ReductionRecipe, label: str
    ) -> ReductionRecipe:
        if not ReductionPlanner._addition_allowed(left.target, right.target, label):
            raise AssertionError("unsafe add recipe reached graph construction")
        return ReductionRecipe.for_add(left, right, label=label)

    def _scale(
        self, factor: int, child: ReductionRecipe, label: str
    ) -> ReductionRecipe:
        unit = self._require_unit(factor)
        return ReductionRecipe.for_scale(unit, child, label=label)

    @staticmethod
    def _sub(
        minuend: ReductionRecipe, subtrahend: ReductionRecipe, label: str
    ) -> ReductionRecipe:
        if not ReductionPlanner._sub_allowed(
            minuend.target, subtrahend.target, label
        ):
            raise AssertionError("unsafe sub recipe reached graph construction")
        return ReductionRecipe.for_sub(minuend, subtrahend, label=label)


def _ceil_ratio(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@cache
def _prime_power_blocks(value: int) -> tuple[int, ...]:
    if value < 2:
        return ()
    blocks: list[int] = []
    remaining = value
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor:
            factor = 3 if factor == 2 else factor + 2
            continue
        block = 1
        while remaining % factor == 0:
            remaining //= factor
            block *= factor
        blocks.append(block)
        factor = 3 if factor == 2 else factor + 2
    if remaining > 1:
        blocks.append(remaining)
    return tuple(blocks)


def _ordered_pair(left: Fraction, right: Fraction) -> tuple[Fraction, Fraction]:
    return (left, right) if left <= right else (right, left)


def _addition_pair_key(
    pair: tuple[Fraction, Fraction],
) -> tuple[int, int, int, int, int, int]:
    left, right = pair
    maximum = max(left.denominator, right.denominator)
    return (
        maximum,
        left.denominator + right.denominator,
        left.denominator,
        right.denominator,
        left.numerator,
        right.numerator,
    )


def _unit_lower_cost(denominator: int) -> TopologyCost:
    twos, odd = factor_power_of_two(denominator)
    odd_k = ceil_log3(odd)
    nodes, edges = unit_topology_size(twos, odd_k)
    return TopologyCost(
        nodes,
        edges,
        1,
        exact=odd in (1, 3),
    )
