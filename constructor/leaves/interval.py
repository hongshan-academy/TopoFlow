"""Certified constructions for rational flows in ``[1/3, 1/2]``."""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations

from ..model.arithmetic import ceil_log3
from ..model.certificate import EdgeState, FlowCertificate
from ..model.graph import NodeKind
from ..operators.composition import boundary_flow
from .unit import half_certificate, third_certificate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SplitOp:
    value: int
    parts: int


type SplitState = tuple[tuple[int, int], ...]
type IntervalGroups = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
type ChainResult = tuple[tuple[SplitOp, ...], IntervalGroups]


@dataclass(frozen=True, slots=True)
class IntervalSolution:
    d: int
    operations: tuple[SplitOp, ...]
    group_a: tuple[int, ...]
    group_b: tuple[int, ...]
    group_c: tuple[int, ...]

    @property
    def element_count(self) -> int:
        return (
            len(self.operations)
            + len(_merge_arities(len(self.group_a), has_base=False))
            + len(_merge_arities(len(self.group_b), has_base=True))
            + len(_merge_arities(len(self.group_c), has_base=True))
            + 2
        )


@dataclass(frozen=True, slots=True)
class IntervalConstruction:
    target: Fraction
    certificate: FlowCertificate
    solution: IntervalSolution | None
    optimized: bool
    strategy: str = "endpoint"


@dataclass(frozen=True, slots=True)
class IntervalPlan:
    """A cheap interval blueprint that can be priced before certification."""

    target: Fraction
    solution: IntervalSolution | None
    optimized: bool
    strategy: str

    @property
    def topology_size(self) -> tuple[int, int]:
        if self.solution is not None:
            return _solution_size(self.solution)
        if self.target == Fraction(1, 3):
            return 4, 5
        if self.target == Fraction(1, 2):
            return 4, 4
        raise ValueError("non-endpoint interval plan requires a solution")

    @property
    def node_count(self) -> int:
        return self.topology_size[0]

    @property
    def edge_count(self) -> int:
        return self.topology_size[1]


def plan_interval(
    target: Fraction, *, optimize: bool = False, search_range: int = 0
) -> IntervalPlan:
    """Choose an interval topology without building its flow graph."""
    if not Fraction(1, 3) <= target <= Fraction(1, 2):
        raise ValueError("interval target must satisfy 1/3 <= x <= 1/2")
    if search_range < 0:
        raise ValueError("search_range must be non-negative")
    if target == Fraction(1, 3):
        return IntervalPlan(target, None, False, "endpoint")
    if target == Fraction(1, 2):
        return IntervalPlan(target, None, False, "endpoint")

    direct = _mixed_radix_solution(target)
    searched = _search_solution(target, search_range) if optimize else None
    candidates = [("mixed-radix", direct)] if direct is not None else []
    if searched is not None:
        candidates.append(("bounded-search", searched))
    if candidates:
        strategy, solution = min(candidates, key=lambda item: item[1].element_count)
    else:
        strategy, solution = "compact-binary", _unit_fallback(target)
    optimized = strategy == "bounded-search"
    node_count, edge_count = _solution_size(solution)
    logger.info(
        "I(%d, %d) plan strategy=%s d=%d operations=%d nodes=%d edges=%d",
        target.numerator,
        target.denominator,
        strategy,
        solution.d,
        len(solution.operations),
        node_count,
        edge_count,
    )
    return IntervalPlan(target, solution, optimized, strategy)


def realize_interval(plan: IntervalPlan) -> FlowCertificate:
    """Materialize a previously selected interval blueprint."""
    if plan.target == Fraction(1, 3):
        return third_certificate()
    if plan.target == Fraction(1, 2):
        return half_certificate()
    if plan.solution is None:
        raise ValueError("non-endpoint interval plan requires a solution")
    certificate = _certificate_from_solution(plan.target, plan.solution)
    if (certificate.node_count, certificate.edge_count) != plan.topology_size:
        raise AssertionError("interval blueprint cost disagrees with constructed graph")
    return certificate


def construct_interval(
    target: Fraction, *, optimize: bool = False, search_range: int = 0
) -> IntervalConstruction:
    """Construct and certify a target in the closed interval ``[1/3, 1/2]``."""
    plan = plan_interval(target, optimize=optimize, search_range=search_range)
    certificate = realize_interval(plan)
    if boundary_flow(certificate) != target:
        raise AssertionError("interval construction produced the wrong boundary flow")
    return IntervalConstruction(
        target, certificate, plan.solution, plan.optimized, plan.strategy
    )


def _solution_size(solution: IntervalSolution) -> tuple[int, int]:
    """Return exact total nodes/edges from the split/merge blueprint."""
    arities = [operation.parts for operation in solution.operations]
    arities.extend(_merge_arities(len(solution.group_a), has_base=False))
    arities.extend(_merge_arities(len(solution.group_b), has_base=True))
    arities.extend(_merge_arities(len(solution.group_c), has_base=True))
    two_port = arities.count(2)
    three_port = 2 + arities.count(3)
    internal = two_port + three_port
    edges = (3 * two_port + 4 * three_port + 2) // 2
    return internal + 2, edges


def _mixed_radix_solution(target: Fraction) -> IntervalSolution | None:
    """Find a goal-directed 2/3-radix chain with only one residual child."""
    p, q = target.numerator, target.denominator
    totals = (q - 2 * p, 3 * p - q, 0)
    k = ceil_log3(p)
    best: IntervalSolution | None = None

    def chains(values: tuple[int, int, int], total: int) -> Iterator[ChainResult]:
        for parts in (2, 3):
            if total % parts:
                continue
            child = total // parts
            quotients = (
                values[0] // child,
                values[1] // child,
                values[2] // child,
            )
            remainders = (
                values[0] % child,
                values[1] % child,
                values[2] % child,
            )
            residual_children = sum(remainders) // child
            if sum(remainders) not in (0, child):
                continue
            groups: IntervalGroups = (
                (child,) * quotients[0],
                (child,) * quotients[1],
                (child,) * quotients[2],
            )
            operation = (SplitOp(total, parts),)
            if residual_children == 0:
                yield operation, groups
                continue
            for sub_operations, sub_groups in chains(remainders, child):
                yield (
                    (*operation, *sub_operations),
                    (
                        groups[0] + sub_groups[0],
                        groups[1] + sub_groups[1],
                        groups[2] + sub_groups[2],
                    ),
                )

    for d in _candidate_ds(p, q, k):
        values = (totals[0], totals[1], d - p)
        for operations, groups in chains(values, d):
            if _priority_elements(sorted(groups[1]), values[0]):
                continue
            candidate = IntervalSolution(d, operations, *map(tuple, groups))
            if best is None or candidate.element_count < best.element_count:
                best = candidate
                logger.debug(
                    "I(%d, %d) mixed-radix candidate d=%d radices=%s elements=%d",
                    target.numerator,
                    target.denominator,
                    d,
                    tuple(operation.parts for operation in operations),
                    candidate.element_count,
                )
    return best


def _unit_fallback(target: Fraction) -> IntervalSolution:
    """Guaranteed compact dyadic partition; only boundary-crossing blocks split."""
    p, q = target.numerator, target.denominator
    a = q - 2 * p
    d = 1 << (p - 1).bit_length()
    operations: list[SplitOp] = []
    groups: tuple[list[int], list[int], list[int]] = ([], [], [])

    def partition(start: int, size: int) -> None:
        end = start + size
        if not (start < a < end or start < p < end):
            group = 0 if end <= a else 1 if end <= p else 2
            groups[group].append(size)
            return
        operations.append(SplitOp(size, 2))
        partition(start, size // 2)
        partition(start + size // 2, size // 2)

    partition(0, d)
    while priority := _priority_elements(sorted(groups[1]), a):
        value = priority[0]
        if value == 1:
            raise AssertionError("unit B fragment cannot be refined further")
        groups[1].remove(value)
        groups[1].extend((value // 2, value // 2))
        operations.append(SplitOp(value, 2))
    return IntervalSolution(
        d,
        tuple(operations),
        tuple(groups[0]),
        tuple(groups[1]),
        tuple(groups[2]),
    )


def _search_solution(target: Fraction, search_range: int) -> IntervalSolution | None:
    """Run the bounded fragment search; invalid certificates never become bounds."""
    p, q = target.numerator, target.denominator
    a, b = q - 2 * p, 3 * p - q
    k = ceil_log3(p)
    max_steps = max(1, k + 1 + search_range)
    best: IntervalSolution | None = None
    fallback_count = _unit_fallback(target).element_count

    for d in _candidate_ds(p, q, k):
        examined = 0
        initial = ((d, 1),)
        queue: deque[tuple[SplitState, tuple[SplitOp, ...]]] = deque([(initial, ())])
        seen: set[SplitState] = {initial}
        while queue:
            state, operations = queue.popleft()
            examined += 1
            items = _expand_state(state)
            partition = _find_partition(items, a, b)
            if partition is not None:
                group_a, group_b, group_c = partition
                if not _priority_elements(sorted(group_b), a):
                    candidate = IntervalSolution(
                        d,
                        operations,
                        tuple(group_a),
                        tuple(group_b),
                        tuple(group_c),
                    )
                    threshold = min(
                        fallback_count,
                        candidate.element_count if best is None else best.element_count,
                    )
                    if candidate.element_count <= threshold + search_range:
                        try:
                            _validate_solution(target, candidate)
                        except (AssertionError, ValueError):
                            pass
                        else:
                            if (
                                best is None
                                or candidate.element_count < best.element_count
                            ):
                                best = candidate
            if len(operations) >= max_steps:
                continue
            lower_bound = len(operations) + math.ceil((len(items) - 1) / 2) + 2
            active_bound = (
                fallback_count
                if best is None
                else min(fallback_count, best.element_count)
            )
            if lower_bound > active_bound + search_range:
                continue
            for value, _ in state:
                for parts in (2, 3):
                    next_state = _apply_split(state, value, parts)
                    if next_state is None or next_state in seen:
                        continue
                    seen.add(next_state)
                    queue.append((next_state, (*operations, SplitOp(value, parts))))
        logger.debug(
            "I(%d, %d) bounded-search layer d=%d states=%d best_elements=%s",
            target.numerator,
            target.denominator,
            d,
            examined,
            None if best is None else best.element_count,
        )
    return best


def _candidate_ds(total: int, q: int, k: int) -> tuple[int, ...]:
    bases = [3 ** (k - n) * 2**n for n in range(k + 1)]
    eligible = [value for value in bases if value >= total]
    minimum = min(eligible) if eligible else total
    upper = min(q - 1, 2 * minimum)
    values: set[int] = set()
    power_two = 1
    while power_two <= upper:
        value = power_two
        while value <= upper:
            if value >= total:
                values.add(value)
            value *= 3
        power_two *= 2
    return tuple(sorted(values))


def _apply_split(state: SplitState, value: int, parts: int) -> SplitState | None:
    counts = dict(state)
    if counts.get(value, 0) == 0 or value % parts:
        return None
    counts[value] -= 1
    if not counts[value]:
        del counts[value]
    child = value // parts
    counts[child] = counts.get(child, 0) + parts
    return tuple(sorted(counts.items()))


def _expand_state(state: SplitState) -> list[int]:
    return [value for value, count in state for _ in range(count)]


def _find_partition(
    items: list[int], a: int, b: int
) -> tuple[list[int], list[int], list[int]] | None:
    """Two-dimensional subset-sum with several deterministic reconstructions."""
    total = sum(items)
    if a < 0 or b < 0 or total < a + b:
        return None
    swapped = a > b
    lo, hi = (b, a) if swapped else (a, b)
    mask = (1 << (hi + 1)) - 1

    def forward(keep_history: bool):
        dp = [0] * (lo + 1)
        dp[0] = 1
        history: list[list[int]] | None = [] if keep_history else None
        for value in items:
            old = dp
            if history is not None:
                history.append(old)
            new = old[:]
            if value <= lo:
                for low_sum in range(lo, value - 1, -1):
                    new[low_sum] |= old[low_sum - value]
            if value <= hi:
                for low_sum in range(lo + 1):
                    new[low_sum] |= (old[low_sum] << value) & mask
            dp = new
        return dp, history

    final, _ = forward(False)
    if not (final[lo] >> hi) & 1:
        return None
    _, history = forward(True)
    assert history is not None
    best = None
    best_key = (True, 10**9, 10**9)
    for order in permutations(("skip", "lo", "hi")):
        low_group: list[int] = []
        high_group: list[int] = []
        third_group: list[int] = []
        low_sum, high_sum = lo, hi
        for index in range(len(items) - 1, -1, -1):
            value = items[index]
            old = history[index]
            for action in order:
                if action == "skip" and (old[low_sum] >> high_sum) & 1:
                    third_group.append(value)
                    break
                if (
                    action == "lo"
                    and low_sum >= value
                    and (old[low_sum - value] >> high_sum) & 1
                ):
                    low_group.append(value)
                    low_sum -= value
                    break
                if (
                    action == "hi"
                    and high_sum >= value
                    and (old[low_sum] >> (high_sum - value)) & 1
                ):
                    high_group.append(value)
                    high_sum -= value
                    break
        group_a, group_b = (
            (high_group, low_group) if swapped else (low_group, high_group)
        )
        priority = len(_priority_elements(sorted(group_b), a))
        merge_cost = (
            len(_merge_arities(len(group_a), has_base=False))
            + len(_merge_arities(len(group_b), has_base=True))
            + len(_merge_arities(len(third_group), has_base=True))
        )
        key = (priority > 0, priority, merge_cost)
        if key < best_key:
            best_key = key
            best = (group_a, group_b, third_group)
    return best


def _priority_elements(values: list[int], initial: int) -> list[int]:
    priority: list[int] = []
    for start in range(0, len(values), 2):
        for value in values[start : start + 2]:
            if value > initial:
                priority.append(value)
        initial += sum(values[start : start + 2])
    return priority


def _merge_arities(count: int, *, has_base: bool) -> tuple[int, ...]:
    """Return the converger arities for one deterministic merge chain."""
    if count < 0:
        raise ValueError("merge item count must be non-negative")
    if count == 0 or (not has_base and count == 1):
        return ()
    first_count = min(count, 2 if has_base else 3)
    first_arity = first_count + int(has_base)
    remaining = count - first_count
    return (
        first_arity,
        *(3 for _ in range(remaining // 2)),
        *((2,) if remaining % 2 else ()),
    )


@dataclass(frozen=True, slots=True)
class _Fragment:
    value: int
    source: str


class _CertificateBuilder:
    def __init__(self, denominator: int) -> None:
        self.denominator = denominator
        self.edges: list[tuple[str, str]] = []
        self.weights: list[int] = []
        self.states: list[EdgeState] = []
        self.fixed: set[int] = set()
        self.next_index: dict[NodeKind, int] = dict.fromkeys(
            (NodeKind.S2, NodeKind.S3, NodeKind.C2, NodeKind.C3), 0
        )

    def reserve(self, kind: NodeKind) -> str:
        return self.new_node(kind)

    def new_node(self, kind: NodeKind) -> str:
        index = self.next_index[kind]
        self.next_index[kind] += 1
        return f"{kind.prefix}{index}"

    def add(
        self,
        source: str,
        target: str,
        weight: int,
        state: EdgeState,
        *,
        fixed: bool = False,
    ) -> None:
        self.edges.append((source, target))
        self.weights.append(weight)
        self.states.append(state)
        if fixed:
            self.fixed.add(len(self.edges) - 1)

    def finish(self, root_source: str) -> FlowCertificate:
        edges = tuple(
            (root_source if source == "__ROOT__" else source, target)
            for source, target in self.edges
        )
        return FlowCertificate(
            edges,
            tuple(Fraction(weight, self.denominator) for weight in self.weights),
            tuple(self.states),
            frozenset(self.fixed),
        )


def _certificate_from_solution(
    target: Fraction, solution: IntervalSolution
) -> FlowCertificate:
    _validate_solution(target, solution)
    p, q = target.numerator, target.denominator
    a = q - 2 * p
    builder = _CertificateBuilder(q)
    main_c = builder.reserve(NodeKind.C3)
    main_s = builder.reserve(NodeKind.S3)
    builder.add("In", main_c, p, EdgeState.FULL)
    builder.add(main_c, main_s, q, EdgeState.FULL, fixed=True)
    builder.add(main_s, "Out", p, EdgeState.EMPTY)

    pool = [_Fragment(solution.d, "__ROOT__")]
    for operation in solution.operations:
        match = next(
            (
                index
                for index, fragment in enumerate(pool)
                if fragment.value == operation.value
            ),
            None,
        )
        if match is None or operation.value % operation.parts:
            raise ValueError("split operation does not match the fragment pool")
        fragment = pool.pop(match)
        kind = NodeKind.S2 if operation.parts == 2 else NodeKind.S3
        splitter = builder.new_node(kind)
        builder.add(fragment.source, splitter, fragment.value, EdgeState.EMPTY)
        child = fragment.value // operation.parts
        pool.extend(_Fragment(child, splitter) for _ in range(operation.parts))

    groups = _assign_fragments(pool, solution)
    _build_merge_chain(
        builder,
        groups[0],
        destination=main_c,
        base=None,
        chain_state=EdgeState.EMPTY,
    )
    _build_merge_chain(
        builder,
        groups[1],
        destination=main_c,
        base=(main_s, a, EdgeState.FULL),
        chain_state=EdgeState.FULL,
    )
    if groups[2]:
        root_source, total = _build_merge_chain(
            builder,
            groups[2],
            destination=None,
            base=(main_s, p, EdgeState.EMPTY),
            chain_state=EdgeState.EMPTY,
        )
        if total != solution.d:
            raise AssertionError("C chain did not reconstruct d")
    else:
        if solution.d != p:
            raise AssertionError("empty C group requires d=p")
        root_source = main_s
    certificate = builder.finish(root_source)
    errors = certificate.validation_errors()
    if errors:
        raise ValueError(f"generated interval certificate is invalid: {errors}")
    return certificate


def _validate_solution(target: Fraction, solution: IntervalSolution) -> None:
    """Validate an interval blueprint arithmetically, without constructing a graph."""
    p, q = target.numerator, target.denominator
    a, b = q - 2 * p, 3 * p - q
    if not p <= solution.d < q:
        raise ValueError("interval root is outside the admissible range")
    if sum(solution.group_a) != a or sum(solution.group_b) != b:
        raise ValueError("interval groups have incorrect target sums")
    if sum(solution.group_c) != solution.d - p:
        raise ValueError("interval C group has incorrect remainder")
    if _priority_elements(sorted(solution.group_b), a):
        raise ValueError("interval B chain contains a priority merge")
    pool = [solution.d]
    for operation in solution.operations:
        try:
            index = pool.index(operation.value)
        except ValueError as exc:
            raise ValueError(
                "split operation does not match the fragment pool"
            ) from exc
        if operation.parts not in (2, 3) or operation.value % operation.parts:
            raise ValueError("invalid interval split operation")
        pool.pop(index)
        pool.extend([operation.value // operation.parts] * operation.parts)
    expected = Counter((*solution.group_a, *solution.group_b, *solution.group_c))
    if Counter(pool) != expected:
        raise ValueError("interval groups do not match the final fragment pool")


def _assign_fragments(
    pool: list[_Fragment], solution: IntervalSolution
) -> tuple[list[_Fragment], list[_Fragment], list[_Fragment]]:
    counters = [
        Counter(solution.group_a),
        Counter(solution.group_b),
        Counter(solution.group_c),
    ]
    groups: tuple[list[_Fragment], list[_Fragment], list[_Fragment]] = ([], [], [])
    for fragment in pool:
        for counter, group in zip(counters, groups, strict=True):
            if counter[fragment.value]:
                counter[fragment.value] -= 1
                group.append(fragment)
                break
        else:
            raise ValueError(f"unassigned fragment of weight {fragment.value}")
    if any(any(counter.values()) for counter in counters):
        raise ValueError("requested groups do not match the final fragment pool")
    return groups


def _build_merge_chain(
    builder: _CertificateBuilder,
    fragments: list[_Fragment],
    *,
    destination: str | None,
    base: tuple[str, int, EdgeState] | None,
    chain_state: EdgeState,
) -> tuple[str, int]:
    fragments = sorted(fragments, key=lambda item: item.value)
    arities = _merge_arities(len(fragments), has_base=base is not None)
    if not arities and base is None and len(fragments) == 1:
        fragment = fragments[0]
        if destination is None:
            return fragment.source, fragment.value
        builder.add(fragment.source, destination, fragment.value, EdgeState.EMPTY)
        return destination, fragment.value
    if not arities:
        raise ValueError("merge chain needs at least one fragment")

    node: str | None = None
    total = 0
    offset = 0
    for index, arity in enumerate(arities):
        next_node = builder.new_node(NodeKind.C2 if arity == 2 else NodeKind.C3)
        carried_inputs = int(index > 0 or (index == 0 and base is not None))
        batch_size = arity - carried_inputs
        batch = fragments[offset : offset + batch_size]
        if len(batch) != batch_size:
            raise AssertionError("merge arity does not match fragment count")
        offset += batch_size
        if index == 0 and base is not None:
            source, value, state = base
            builder.add(source, next_node, value, state)
            total += value
        elif node is not None:
            builder.add(node, next_node, total, chain_state)
        for fragment in batch:
            builder.add(fragment.source, next_node, fragment.value, EdgeState.EMPTY)
            total += fragment.value
        node = next_node
    if offset != len(fragments) or node is None:
        raise AssertionError("merge chain did not consume every fragment")
    if destination is not None:
        builder.add(node, destination, total, chain_state)
    return node, total
