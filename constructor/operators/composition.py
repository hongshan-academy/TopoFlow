"""Exact addition, multiplication, and restricted-sub operators."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from ..model.certificate import EdgeState, FlowCertificate
from graph import NodeKind


def boundary_flow(certificate: FlowCertificate) -> Fraction:
    """Return the common flow on a one-port module's source and sink edges."""
    source = [
        certificate.flows[index]
        for index, (left, _) in enumerate(certificate.edges)
        if left == "In"
    ]
    sink = [
        certificate.flows[index]
        for index, (_, right) in enumerate(certificate.edges)
        if right == "Out"
    ]
    if len(source) != 1 or len(sink) != 1 or source[0] != sink[0]:
        raise ValueError(
            "certificate must have one equally-valued source and sink edge"
        )
    return source[0]


def full_certificate() -> FlowCertificate:
    """Return the one-edge primitive carrying fixed full flow."""
    return FlowCertificate(
        (("In", "Out"),),
        (Fraction(1),),
        (EdgeState.FULL,),
        frozenset({0}),
    )


def replace_fixed_edge(
    outer: FlowCertificate, inner: FlowCertificate
) -> FlowCertificate:
    """Replace the sole fixed edge of ``outer`` by ``inner``, multiplying flows."""
    _require_valid(outer, "outer")
    _require_valid(inner, "inner")
    if len(outer.fixed) != 1:
        raise ValueError("outer certificate must have exactly one fixed edge")
    removed = next(iter(outer.fixed))
    left, right = outer.edges[removed]
    if not (left.startswith(("C2_", "C3_")) and right.startswith(("S2_", "S3_"))):
        raise ValueError(
            "replaceable fixed edge must point from a converger to a splitter"
        )

    scale = boundary_flow(inner)
    mapping = _fresh_internal_names(outer, inner)
    edges: list[tuple[str, str]] = []
    flows: list[Fraction] = []
    states: list[EdgeState] = []
    fixed: set[int] = set()
    for index, edge in enumerate(outer.edges):
        if index == removed:
            continue
        edges.append(edge)
        flows.append(outer.flows[index] * scale)
        states.append(outer.states[index])
    for index, (source, target) in enumerate(inner.edges):
        edges.append(
            (
                left if source == "In" else mapping[source],
                right if target == "Out" else mapping[target],
            )
        )
        flows.append(inner.flows[index])
        states.append(inner.states[index])
        if index in inner.fixed:
            fixed.add(len(edges) - 1)
    result = FlowCertificate(
        tuple(edges), tuple(flows), tuple(states), frozenset(fixed)
    )
    _require_valid(result, "product")
    if boundary_flow(result) != boundary_flow(outer) * scale:
        raise AssertionError("full-edge replacement produced the wrong boundary flow")
    return result


def parallel_sum(left: FlowCertificate, right: FlowCertificate) -> FlowCertificate:
    """Put two modules in parallel, realizing ``x + y`` when it is below one."""
    _require_valid(left, "left addend")
    _require_valid(right, "right addend")
    total = boundary_flow(left) + boundary_flow(right)
    if total >= 1:
        raise ValueError("parallel sum requires x + y < 1")

    counters: defaultdict[NodeKind, int] = defaultdict(int)
    splitter = _allocate(NodeKind.S2, counters)
    converger = _allocate(NodeKind.C2, counters)
    edges: list[tuple[str, str]] = [("In", splitter)]
    flows: list[Fraction] = [total]
    states: list[EdgeState] = [EdgeState.FULL]
    fixed: set[int] = set()
    for certificate in (left, right):
        names = {
            node: _allocate(NodeKind.from_node(node), counters)
            for node in sorted(certificate.graph.nodes - {"In", "Out"})
        }
        for index, (source, target) in enumerate(certificate.edges):
            edges.append(
                (
                    splitter if source == "In" else names[source],
                    converger if target == "Out" else names[target],
                )
            )
            flows.append(certificate.flows[index])
            states.append(certificate.states[index])
            if index in certificate.fixed:
                fixed.add(len(edges) - 1)
    edges.append((converger, "Out"))
    flows.append(total)
    states.append(EdgeState.EMPTY)
    result = FlowCertificate(
        tuple(edges), tuple(flows), tuple(states), frozenset(fixed)
    )
    _require_valid(result, "parallel sum")
    if boundary_flow(result) != total:
        raise AssertionError("parallel sum produced the wrong boundary flow")
    return result


def restricted_sub(
    minuend: FlowCertificate, subtrahend: FlowCertificate
) -> FlowCertificate:
    """Construct ``a-b`` for the non-degenerate restricted case ``a > 2b``.

    Equality is algebraically valid but always dominated by returning the
    subtrahend graph itself, because ``a-b=b`` when ``a=2b``.
    """
    _require_valid(minuend, "minuend")
    _require_valid(subtrahend, "subtrahend")
    a = boundary_flow(minuend)
    b = boundary_flow(subtrahend)
    if a <= 2 * b:
        raise ValueError("restricted sub requires a > 2b")

    counters: defaultdict[NodeKind, int] = defaultdict(int)
    converger = _allocate(NodeKind.C2, counters)
    splitter = _allocate(NodeKind.S2, counters)
    difference = a - b
    edges: list[tuple[str, str]] = [("In", converger)]
    flows: list[Fraction] = [difference]
    states: list[EdgeState] = [EdgeState.FULL]
    fixed: set[int] = set()

    for certificate, source_boundary, target_boundary in (
        (minuend, converger, splitter),
        (subtrahend, splitter, converger),
    ):
        names = {
            node: _allocate(NodeKind.from_node(node), counters)
            for node in sorted(certificate.graph.nodes - {"In", "Out"})
        }
        for index, (source, target) in enumerate(certificate.edges):
            edges.append(
                (
                    source_boundary if source == "In" else names[source],
                    target_boundary if target == "Out" else names[target],
                )
            )
            flows.append(certificate.flows[index])
            states.append(certificate.states[index])
            if index in certificate.fixed:
                fixed.add(len(edges) - 1)

    edges.append((splitter, "Out"))
    flows.append(difference)
    states.append(EdgeState.EMPTY)
    result = FlowCertificate(
        tuple(edges), tuple(flows), tuple(states), frozenset(fixed)
    )
    _require_valid(result, "restricted sub")
    if boundary_flow(result) != difference:
        raise AssertionError("restricted sub produced the wrong boundary flow")
    return result


def _fresh_internal_names(
    outer: FlowCertificate, inner: FlowCertificate
) -> dict[str, str]:
    next_index: defaultdict[NodeKind, int] = defaultdict(int)
    for node in outer.graph.nodes - {"In", "Out"}:
        kind = NodeKind.from_node(node)
        next_index[kind] = max(next_index[kind], int(node.rsplit("_", 1)[1]) + 1)
    mapping: dict[str, str] = {}
    for node in sorted(inner.graph.nodes - {"In", "Out"}):
        kind = NodeKind.from_node(node)
        mapping[node] = f"{kind.prefix}{next_index[kind]}"
        next_index[kind] += 1
    return mapping


def _allocate(kind: NodeKind, counters: defaultdict[NodeKind, int]) -> str:
    node = f"{kind.prefix}{counters[kind]}"
    counters[kind] += 1
    return node


def _require_valid(certificate: FlowCertificate, name: str) -> None:
    errors = certificate.validation_errors()
    if errors:
        raise ValueError(f"{name} certificate is invalid: {errors}")
