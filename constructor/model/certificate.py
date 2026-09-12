"""Exact flow/state certificates for completed TopoFlow graphs."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from time import perf_counter

from .graph import Edge, Graph

logger = logging.getLogger(__name__)


class EdgeState(Enum):
    EMPTY = (True, False)
    FULL = (False, True)
    PARTIAL = (True, True)

    def __init__(self, can_in: bool, can_out: bool) -> None:
        self.can_in = can_in
        self.can_out = can_out


@dataclass(frozen=True, slots=True)
class CertificateCheck:
    errors: tuple[str, ...]
    rank: int
    edge_count: int

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def full_rank(self) -> bool:
        return self.rank == self.edge_count


@dataclass(frozen=True, slots=True)
class FlowCertificate:
    edges: tuple[Edge, ...]
    flows: tuple[Fraction, ...]
    states: tuple[EdgeState, ...]
    fixed: frozenset[int]
    _graph: Graph = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        count = len(self.edges)
        if len(self.flows) != count or len(self.states) != count:
            raise ValueError("edges, flows, and states must have equal lengths")
        if any(index < 0 or index >= count for index in self.fixed):
            raise ValueError("fixed edge index is out of range")
        object.__setattr__(self, "_graph", Graph.from_edges(self.edges))

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def node_count(self) -> int:
        return len(self._graph.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def validation_errors(self) -> tuple[str, ...]:
        """Check graph, flow, state, and local rules without solving a rank system."""
        incoming, outgoing = _incidence(self.graph, self.edges)
        return self._validation_errors(incoming, outgoing, _parallel_edges(self.edges))

    def _validation_errors(
        self,
        incoming: dict[str, list[int]],
        outgoing: dict[str, list[int]],
        parallel: dict[Edge, list[int]],
    ) -> tuple[str, ...]:
        errors: list[str] = list(self.graph.validation_errors())
        for index, flow in enumerate(self.flows):
            if not 0 < flow <= 1:
                errors.append(f"edge {index} has flow outside (0, 1]: {flow}")
        if not self.fixed:
            errors.append("certificate must contain at least one fixed edge")
        for index, edge in enumerate(self.edges):
            candidate = _is_fixed_candidate(edge)
            if index in self.fixed:
                if self.flows[index] != 1:
                    errors.append(f"fixed edge {index} does not have flow 1")
            elif candidate and self.flows[index] >= 1:
                errors.append(f"non-fixed candidate edge {index} is not below 1")
            source, target = edge
            state = self.states[index]
            if state is EdgeState.PARTIAL and not (
                _is_splitter(source) and _is_converger(target)
            ):
                errors.append(
                    f"edge {index} is PARTIAL but is not splitter-to-converger"
                )
            if source == "In" and state is not EdgeState.FULL:
                errors.append(f"source edge {index} must be FULL")
            if (
                target == "Out"
                and index not in self.fixed
                and state is not EdgeState.EMPTY
            ):
                errors.append(f"non-fixed sink edge {index} must be EMPTY")

        for edge, indices in parallel.items():
            first = self.flows[indices[0]]
            if any(self.flows[index] != first for index in indices[1:]):
                errors.append(f"parallel edges {edge!r} do not have equal flow")

        for node in sorted(self.graph.nodes - {"In", "Out"}):
            in_total = sum((self.flows[i] for i in incoming[node]), Fraction())
            out_total = sum((self.flows[i] for i in outgoing[node]), Fraction())
            if in_total != out_total:
                errors.append(
                    f"flow is not conserved at {node}: {in_total} != {out_total}"
                )
            if _is_splitter(node):
                input_index = incoming[node][0]
                for index in outgoing[node]:
                    if index in self.fixed or input_index in self.fixed:
                        continue
                    if (
                        self.states[index].can_in
                        and not self.states[input_index].can_in
                    ):
                        errors.append(
                            f"can_in propagation fails at {node} on edge {index}"
                        )
                non_fixed = [i for i in outgoing[node] if i not in self.fixed]
                for index in non_fixed:
                    if self.states[index].can_in and any(
                        self.flows[index] < self.flows[other] for other in non_fixed
                    ):
                        errors.append(
                            f"edge {index} is not locally maximal at splitter {node}"
                        )
            elif _is_converger(node):
                output_index = outgoing[node][0]
                for index in incoming[node]:
                    if index in self.fixed or output_index in self.fixed:
                        continue
                    if (
                        self.states[index].can_out
                        and not self.states[output_index].can_out
                    ):
                        errors.append(
                            f"can_out propagation fails at {node} on edge {index}"
                        )
                non_fixed = [i for i in incoming[node] if i not in self.fixed]
                for index in non_fixed:
                    if self.states[index].can_out and any(
                        self.flows[index] < self.flows[other] for other in non_fixed
                    ):
                        errors.append(
                            f"edge {index} is not locally maximal at converger {node}"
                        )

        return tuple(errors)

    def check(self, *, require_full_rank: bool = True) -> CertificateCheck:
        started = perf_counter()
        incoming, outgoing = _incidence(self.graph, self.edges)
        parallel = _parallel_edges(self.edges)
        errors = list(self._validation_errors(incoming, outgoing, parallel))
        rows = _equation_rows(self, incoming, outgoing, parallel)
        rank = _matrix_rank(rows)
        if require_full_rank and rank != len(self.edges):
            errors.append(
                f"certificate equations have rank {rank}, expected {len(self.edges)}"
            )
        result = CertificateCheck(tuple(errors), rank, len(self.edges))
        logger.debug(
            "certificate check edges=%d rows=%d rank=%d valid=%s elapsed=%.3fs",
            len(self.edges),
            len(rows),
            rank,
            result.valid,
            perf_counter() - started,
        )
        return result


def _incidence(
    graph: Graph, edges: tuple[Edge, ...]
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    incoming: dict[str, list[int]] = {node: [] for node in graph.nodes}
    outgoing: dict[str, list[int]] = {node: [] for node in graph.nodes}
    for index, (source, target) in enumerate(edges):
        outgoing[source].append(index)
        incoming[target].append(index)
    return incoming, outgoing


def _parallel_edges(edges: tuple[Edge, ...]) -> dict[Edge, list[int]]:
    parallel: dict[Edge, list[int]] = {}
    for index, edge in enumerate(edges):
        parallel.setdefault(edge, []).append(index)
    return parallel


def _is_splitter(node: str) -> bool:
    return node.startswith(("S2_", "S3_"))


def _is_converger(node: str) -> bool:
    return node.startswith(("C2_", "C3_"))


def _is_fixed_candidate(edge: Edge) -> bool:
    source, target = edge
    return (source == "In" or _is_converger(source)) and (
        target == "Out" or _is_splitter(target)
    )


def _equation_rows(
    certificate: FlowCertificate,
    incoming: dict[str, list[int]],
    outgoing: dict[str, list[int]],
    parallel: dict[Edge, list[int]],
) -> list[list[Fraction]]:
    count = len(certificate.edges)
    rows: list[list[Fraction]] = []

    def relation(left: int, right: int) -> None:
        row = [Fraction() for _ in range(count)]
        row[left], row[right] = Fraction(1), Fraction(-1)
        rows.append(row)

    for node in sorted(certificate.graph.nodes - {"In", "Out"}):
        row = [Fraction() for _ in range(count)]
        for index in incoming[node]:
            row[index] += 1
        for index in outgoing[node]:
            row[index] -= 1
        rows.append(row)
    for indices in parallel.values():
        for index in indices[1:]:
            relation(indices[0], index)
    for index in sorted(certificate.fixed):
        row = [Fraction() for _ in range(count)]
        row[index] = Fraction(1)
        rows.append(row)
    for node in sorted(certificate.graph.nodes - {"In", "Out"}):
        if _is_splitter(node):
            selected = [
                i
                for i in outgoing[node]
                if i not in certificate.fixed and certificate.states[i].can_in
            ]
        elif _is_converger(node):
            selected = [
                i
                for i in incoming[node]
                if i not in certificate.fixed and certificate.states[i].can_out
            ]
        else:
            continue
        for index in selected[1:]:
            relation(selected[0], index)
    return rows


def _matrix_rank(rows: Iterable[list[Fraction]]) -> int:
    matrix = [row[:] for row in rows if any(row)]
    if not matrix:
        return 0
    columns = len(matrix[0])
    rank = 0
    for column in range(columns):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        value = matrix[rank][column]
        matrix[rank] = [entry / value for entry in matrix[rank]]
        for index, row in enumerate(matrix):
            if index == rank or not row[column]:
                continue
            factor = row[column]
            matrix[index] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(row, matrix[rank], strict=True)
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank
