#!/usr/bin/env python3
"""Adapt TopoFlow GA results to the minimal logistics layout solver.

TopoFlow chooses a logical splitter/merger topology.  This module keeps that
topology unchanged, maps its internal nodes to Endfield logistics facilities,
and delegates physical placement and internal routing to the adjacent
``solver.py`` standalone solver.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from . import solver as solver_module


SCHEMA = "topoflow-logistics/v1"
SOLUTION_SCHEMA = "topoflow-logistics-solution/v1"
LAYOUT_REQUEST_SCHEMA = "topoflow-logistics-layout/v1"
# GA/editor exports may append a numeric uniqueness suffix, for example
# ``C2_0__45131``.  The suffix is identity metadata only; the first capture
# group still determines the facility family and therefore its degree rules.
NODE_PATTERN = re.compile(r"^(S2|S3|C2|C3)_\d+(?:__\d+)?$")
EXPECTED_DEGREES = {
    "S2": (1, 2),
    "S3": (1, 3),
    "C2": (2, 1),
    "C3": (3, 1),
}
FAMILY_BY_DEGREE = {degree: family for family, degree in EXPECTED_DEGREES.items()}
TRANSPORT_KINDS = {"belt": 1, "pipe": 2}
TERMINAL_DIRECTIONS = {"bottom": 0, "right": 1, "top": 2, "left": 3}
LayoutHintCallback = Callable[[dict[str, object]], None]
LAYOUT_HINT_SCHEMA = "topoflow-facility-layout-hint/v1"
SEARCH_MODES = {"fast", "balanced", "optimal"}


@dataclass(frozen=True)
class TopoFlowEdge:
    edge_id: str
    source: str
    target: str
    source_index: int


@dataclass(frozen=True)
class TerminalAnchor:
    side: str
    offset: int


@dataclass(frozen=True)
class SearchExpansion:
    enabled: bool
    step: int
    max_rows: int
    max_columns: int


@dataclass(frozen=True)
class AdaptedTopoFlowProblem:
    problem: solver_module.Problem
    node_to_facility: dict[str, int]
    node_families: dict[str, str]
    internal_edges: tuple[TopoFlowEdge, ...]
    external_edges: tuple[TopoFlowEdge, ...]
    graph: dict[str, object]
    selection: dict[str, object]
    transport: str
    require_splitter_merger_belt_cell: bool
    edge_min_transit_cells: dict[str, int]
    terminal_anchors: dict[str, TerminalAnchor]
    search_expansion: SearchExpansion


class SolveFailure(RuntimeError):
    def __init__(self, label: str, *, retryable: bool) -> None:
        super().__init__(label)
        self.label = label
        self.retryable = retryable


@dataclass(frozen=True)
class NodeTypeMismatch:
    node: str
    declared_family: str
    inferred_family: str
    expected_degree: tuple[int, int]
    actual_degree: tuple[int, int]


class NodeTypeMismatchError(ValueError):
    def __init__(self, mismatches: list[NodeTypeMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        first = mismatches[0]
        super().__init__(
            f"{first.node} must have degree {first.expected_degree}, "
            f"got {first.actual_degree}; {len(mismatches)} node type(s) can be inferred from degree"
        )


def _select_result(raw: object, rank: int) -> tuple[dict[str, object], dict[str, object]]:
    if rank < 1:
        raise ValueError("rank must be at least 1")

    selected: dict[str, object]
    if isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, dict)]
        if len(candidates) != len(raw) or not candidates:
            raise ValueError("TopoFlow result array must contain non-empty objects")
        exact = [item for item in candidates if item.get("rank") == rank]
        if exact:
            selected = exact[0]
        elif all("rank" not in item for item in candidates) and rank <= len(candidates):
            selected = candidates[rank - 1]
        else:
            available = [item.get("rank") for item in candidates if "rank" in item]
            raise ValueError(f"TopoFlow rank {rank} not found; available ranks: {available}")
    elif isinstance(raw, dict):
        selected = raw
    else:
        raise ValueError("TopoFlow input must be a result object or result array")

    graph_value = selected.get("graph", selected)
    if not isinstance(graph_value, dict):
        raise ValueError("selected TopoFlow result must contain a graph object")
    graph = graph_value
    return selected, graph


def _parse_graph(graph: dict[str, object]) -> tuple[list[str], list[tuple[str, str]]]:
    raw_nodes = graph.get("nodes")
    raw_edges = graph.get("edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError("graph.edges must be a non-empty list")

    edges: list[tuple[str, str]] = []
    inferred_nodes: list[str] = []
    inferred_seen: set[str] = set()
    for index, value in enumerate(raw_edges):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"graph.edges[{index}] must be [source, target]")
        source, target = value
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"graph.edges[{index}] endpoints must be strings")
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"graph.edges[{index}] endpoints must be non-empty strings")
        if source == target:
            raise ValueError(f"graph.edges[{index}] is a self-loop on {source!r}")
        edges.append((source, target))
        for node in (source, target):
            if node not in inferred_seen:
                inferred_seen.add(node)
                inferred_nodes.append(node)

    if isinstance(raw_nodes, list):
        if not raw_nodes:
            raise ValueError("graph.nodes must be a non-empty list")
        nodes: list[str] = []
        seen_nodes: set[str] = set()
        for index, value in enumerate(raw_nodes):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"graph.nodes[{index}] must be a non-empty string")
            node = value.strip()
            if node in seen_nodes:
                raise ValueError(f"graph.nodes contains duplicate node ID {node!r}")
            seen_nodes.add(node)
            nodes.append(node)
        for index, (source, target) in enumerate(edges):
            if source not in seen_nodes or target not in seen_nodes:
                raise ValueError(f"graph.edges[{index}] references an unknown node")
    elif isinstance(raw_nodes, int) and not isinstance(raw_nodes, bool):
        if raw_nodes < 1:
            raise ValueError("graph.nodes must be a positive integer or a non-empty list")
        nodes = inferred_nodes
        if raw_nodes != len(nodes):
            raise ValueError(
                f"graph.nodes declares {raw_nodes} nodes, but graph.edges contain "
                f"{len(nodes)} unique node IDs"
            )
        for field, actual in (("verified_nodes", len(nodes)), ("edges_count", len(edges))):
            if field not in graph:
                continue
            declared = graph[field]
            if not isinstance(declared, int) or isinstance(declared, bool):
                raise ValueError(f"graph.{field} must be an integer")
            if declared != actual:
                raise ValueError(f"graph.{field} declares {declared}, but the actual count is {actual}")
    else:
        raise ValueError("graph.nodes must be a positive integer or a non-empty list")

    return nodes, edges


def _node_family(node: str) -> str:
    match = NODE_PATTERN.fullmatch(node)
    if match is None:
        raise ValueError(
            f"unsupported TopoFlow node {node!r}; expected In, Out, "
            "S2_<n>, S3_<n>, C2_<n>, or C3_<n>, optionally followed by __<n>"
        )
    return match.group(1)


def _validate_topology(
    nodes: list[str],
    edges: list[tuple[str, str]],
    *,
    infer_types_from_degrees: bool = False,
) -> dict[str, str]:
    if "In" not in nodes or "Out" not in nodes:
        raise ValueError("TopoFlow graph must contain canonical terminal nodes 'In' and 'Out'")

    degrees = {node: [0, 0] for node in nodes}
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    reverse_adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        degrees[source][1] += 1
        degrees[target][0] += 1
        adjacency[source].append(target)
        reverse_adjacency[target].append(source)

    if tuple(degrees["In"]) != (0, 1):
        raise ValueError(f"In must have degree (0, 1), got {tuple(degrees['In'])}")
    if tuple(degrees["Out"]) != (1, 0):
        raise ValueError(f"Out must have degree (1, 0), got {tuple(degrees['Out'])}")

    families: dict[str, str] = {}
    mismatches: list[NodeTypeMismatch] = []
    for node in nodes:
        if node in ("In", "Out"):
            continue
        declared_family = _node_family(node)
        expected = EXPECTED_DEGREES[declared_family]
        actual = (degrees[node][0], degrees[node][1])
        inferred_family = FAMILY_BY_DEGREE.get(actual)
        if inferred_family is None:
            raise ValueError(
                f"{node} has unsupported degree {actual}; expected one of "
                f"{sorted(FAMILY_BY_DEGREE)}"
            )
        if actual != expected:
            mismatches.append(
                NodeTypeMismatch(
                    node=node,
                    declared_family=declared_family,
                    inferred_family=inferred_family,
                    expected_degree=expected,
                    actual_degree=actual,
                )
            )
        families[node] = inferred_family if infer_types_from_degrees else declared_family

    if mismatches and not infer_types_from_degrees:
        raise NodeTypeMismatchError(mismatches)

    def reachable(start: str, links: dict[str, list[str]]) -> set[str]:
        visited: set[str] = set()
        pending = [start]
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(links[node])
        return visited

    from_source = reachable("In", adjacency)
    to_sink = reachable("Out", reverse_adjacency)
    disconnected = [node for node in nodes if node not in from_source or node not in to_sink]
    if disconnected:
        raise ValueError(f"all TopoFlow nodes must lie on an In-to-Out path; invalid: {disconnected}")
    return families


def _facility_for(family: str, kind: int) -> dict[str, object]:
    positive = str(kind)
    negative = str(-kind)
    if family.startswith("S"):
        return {
            "type": "logi",
            "size": [1, 1],
            positive: [[2, 0]],
            negative: [[0, 0], [1, 0], [3, 0]],
        }
    return {
        "type": "logi",
        "size": [1, 1],
        positive: [[0, 0], [1, 0], [3, 0]],
        negative: [[2, 0]],
    }


def _parse_terminal_anchors(
    raw: dict[str, object] | None,
    rows: int,
    columns: int,
) -> dict[str, TerminalAnchor]:
    if raw is None:
        return {}
    if set(raw) != {"In", "Out"}:
        raise ValueError("terminals must contain exactly 'In' and 'Out'")

    anchors: dict[str, TerminalAnchor] = {}
    for terminal in ("In", "Out"):
        value = raw[terminal]
        if not isinstance(value, dict):
            raise ValueError(f"terminals.{terminal} must be an object")
        side = value.get("side")
        offset = value.get("offset")
        if not isinstance(side, str) or side not in TERMINAL_DIRECTIONS:
            raise ValueError(
                f"terminals.{terminal}.side must be left, right, top, or bottom"
            )
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError(f"terminals.{terminal}.offset must be an integer")
        limit = rows if side in ("left", "right") else columns
        if not 0 <= offset < limit:
            raise ValueError(
                f"terminals.{terminal}.offset must be between 0 and {limit - 1} for side {side}"
            )
        anchors[terminal] = TerminalAnchor(side=side, offset=offset)
    return anchors


def _parse_search_expansion(
    *,
    enabled: bool,
    step: int,
    max_rows: int | None,
    max_columns: int | None,
    rows: int,
    columns: int,
) -> SearchExpansion:
    if not isinstance(enabled, bool):
        raise ValueError("auto_expand must be a boolean")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("expand_step must be a positive integer")
    resolved_max_rows = max_rows if max_rows is not None else rows + (8 if enabled else 0)
    resolved_max_columns = max_columns if max_columns is not None else columns + (8 if enabled else 0)
    if not isinstance(resolved_max_rows, int) or isinstance(resolved_max_rows, bool):
        raise ValueError("max_rows must be an integer")
    if not isinstance(resolved_max_columns, int) or isinstance(resolved_max_columns, bool):
        raise ValueError("max_columns must be an integer")
    if resolved_max_rows < rows or resolved_max_columns < columns:
        raise ValueError("maximum expansion grid cannot be smaller than the initial grid")
    return SearchExpansion(
        enabled=enabled,
        step=step,
        max_rows=resolved_max_rows,
        max_columns=resolved_max_columns,
    )


def adapt_topoflow(
    raw: object,
    *,
    rank: int = 1,
    rows: int = 12,
    columns: int = 12,
    transport: str = "belt",
    require_splitter_merger_belt_cell: bool = False,
    edge_min_transit_cells: dict[str, int] | None = None,
    name: str | None = None,
    terminal_anchors: dict[str, object] | None = None,
    auto_expand: bool = False,
    expand_step: int = 1,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> AdaptedTopoFlowProblem:
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if transport not in TRANSPORT_KINDS:
        raise ValueError("transport must be 'belt' or 'pipe'")
    if not isinstance(require_splitter_merger_belt_cell, bool):
        raise ValueError("require_splitter_merger_belt_cell must be a boolean")
    anchors = _parse_terminal_anchors(terminal_anchors, rows, columns)
    expansion = _parse_search_expansion(
        enabled=auto_expand,
        step=expand_step,
        max_rows=max_rows,
        max_columns=max_columns,
        rows=rows,
        columns=columns,
    )

    selected, graph_value = _select_result(raw, rank)
    nodes, edges = _parse_graph(graph_value)
    infer_types_from_degrees = graph_value.get("inferNodeTypesFromDegrees", False)
    if not isinstance(infer_types_from_degrees, bool):
        raise ValueError("graph.inferNodeTypesFromDegrees must be a boolean")
    families = _validate_topology(
        nodes,
        edges,
        infer_types_from_degrees=infer_types_from_degrees,
    )
    internal_nodes = [node for node in nodes if node not in ("In", "Out")]
    if not internal_nodes:
        raise ValueError("TopoFlow graph has no splitter or merger facilities to place")
    initial_grid_is_too_small = len(internal_nodes) > rows * columns
    if initial_grid_is_too_small and not expansion.enabled:
        raise ValueError("the requested grid has fewer cells than TopoFlow internal nodes")
    if len(internal_nodes) > expansion.max_rows * expansion.max_columns:
        raise ValueError(
            "the maximum expansion grid has fewer cells than TopoFlow internal nodes"
        )

    kind = TRANSPORT_KINDS[transport]
    node_to_facility = {node: index for index, node in enumerate(internal_nodes)}
    facilities = [_facility_for(families[node], kind) for node in internal_nodes]
    internal_edges: list[TopoFlowEdge] = []
    external_edges: list[TopoFlowEdge] = []
    raw_connections: list[list[int]] = []
    usage: dict[tuple[str, int], int] = {}

    for source_index, (source, target) in enumerate(edges):
        edge = TopoFlowEdge(
            edge_id=f"topoflow-edge-{source_index + 1}",
            source=source,
            target=target,
            source_index=source_index,
        )
        source_internal = source in node_to_facility
        target_internal = target in node_to_facility
        if source_internal:
            usage[(source, -kind)] = usage.get((source, -kind), 0) + 1
        if target_internal:
            usage[(target, kind)] = usage.get((target, kind), 0) + 1

        if source_internal and target_internal:
            internal_edges.append(edge)
            raw_connections.append([kind, node_to_facility[source], node_to_facility[target]])
        elif (source == "In" and target_internal) or (source_internal and target == "Out"):
            external_edges.append(edge)
        else:
            raise ValueError(
                f"graph.edges[{source_index}] must be internal, In->facility, or facility->Out"
            )

    for (node, port_kind), count in usage.items():
        facility = facilities[node_to_facility[node]]
        available_value = facility.get(str(port_kind), [])
        available = len(available_value) if isinstance(available_value, list) else 0
        if count > available:
            raise ValueError(
                f"{node} needs {count} uses of port kind {port_kind}, but the physical facility has {available}"
            )

    resolved_edge_minimums = dict(edge_min_transit_cells or {})
    internal_edge_ids = {edge.edge_id for edge in internal_edges}
    unknown_minimums = set(resolved_edge_minimums).difference(internal_edge_ids)
    if unknown_minimums:
        raise ValueError(
            f"edge minimum constraints reference non-internal edges: {sorted(unknown_minimums)}"
        )
    for edge_id, minimum in resolved_edge_minimums.items():
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum not in (0, 1, 2):
            raise ValueError(f"edge minimum for {edge_id} must be 0, 1, or 2")
    for edge in internal_edges:
        resolved_edge_minimums.setdefault(edge.edge_id, 0)

    selection = {
        key: value
        for key, value in selected.items()
        if key not in ("graph", "nodes", "edges", "nodeTypes", "inferNodeTypesFromDegrees")
    }
    selected_rank = selection.get("rank", rank)
    problem_name = name or f"topoflow_rank_{selected_rank}"
    canonical_graph: dict[str, object] = {
        "nodes": list(nodes),
        "edges": [[source, target] for source, target in edges],
        "nodeTypes": dict(families),
        "inferNodeTypesFromDegrees": infer_types_from_degrees,
    }
    raw_problem: dict[str, object] = {
        "name": problem_name,
        "description": f"TopoFlow rank {selected_rank} physical layout",
        "n": rows,
        "m": columns,
        "mach": facilities,
        "belt": raw_connections,
        "constraints": {
            "require_splitter_merger_belt_cell": require_splitter_merger_belt_cell,
            "edge_min_transit_cells": [
                resolved_edge_minimums[edge.edge_id] for edge in internal_edges
            ],
        },
        "_topoflow": {
            "schema": SCHEMA,
            "selection": selection,
            "graph": canonical_graph,
            "transport": transport,
            "node_to_facility": node_to_facility,
            "terminals": {
                terminal: {"side": anchor.side, "offset": anchor.offset}
                for terminal, anchor in anchors.items()
            },
            "search_expansion": {
                "enabled": expansion.enabled,
                "step": expansion.step,
                "max_rows": expansion.max_rows,
                "max_columns": expansion.max_columns,
            },
        },
    }
    parse_input = raw_problem
    if initial_grid_is_too_small:
        # The core parser normally rejects a structurally undersized grid. Parse
        # against the allowed maximum to validate everything else, then restore
        # the requested size so the first recorded attempt remains truthful.
        parse_input = dict(raw_problem)
        parse_input["n"] = expansion.max_rows
        parse_input["m"] = expansion.max_columns
    problem = solver_module.parse_problem(parse_input, fallback_name=problem_name)
    if initial_grid_is_too_small:
        problem = replace(problem, rows=rows, columns=columns, raw=raw_problem)
    return AdaptedTopoFlowProblem(
        problem=problem,
        node_to_facility=node_to_facility,
        node_families=families,
        internal_edges=tuple(internal_edges),
        external_edges=tuple(external_edges),
        graph=canonical_graph,
        selection=selection,
        transport=transport,
        require_splitter_merger_belt_cell=require_splitter_merger_belt_cell,
        edge_min_transit_cells=resolved_edge_minimums,
        terminal_anchors=anchors,
        search_expansion=expansion,
    )


def adapt_layout_request(
    raw: object,
    *,
    rank: int | None = None,
    rows: int | None = None,
    columns: int | None = None,
    transport: str | None = None,
    require_splitter_merger_belt_cell: bool | None = None,
    edge_min_transit_cells: dict[str, int] | None = None,
    name: str | None = None,
    auto_expand: bool | None = None,
    expand_step: int | None = None,
    max_rows: int | None = None,
    max_columns: int | None = None,
    terminal_anchors: dict[str, object] | None = None,
) -> AdaptedTopoFlowProblem:
    if not isinstance(raw, dict) or raw.get("schema") != LAYOUT_REQUEST_SCHEMA:
        raise ValueError(f"layout request schema must be {LAYOUT_REQUEST_SCHEMA!r}")
    graph_input = raw.get("topoflow")
    if graph_input is None:
        raise ValueError("layout request must contain a topoflow result")
    grid = raw.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("layout request grid must be an object")
    request_rank = raw.get("rank", 1)
    request_rows = grid.get("rows")
    request_columns = grid.get("columns")
    request_transport = raw.get("transport", "belt")
    request_name = raw.get("name")
    terminals = raw.get("terminals")
    expansion = raw.get("searchExpansion", {})
    constraints = raw.get("constraints", {})
    if not isinstance(request_rank, int) or isinstance(request_rank, bool):
        raise ValueError("layout request rank must be an integer")
    if not isinstance(request_rows, int) or isinstance(request_rows, bool):
        raise ValueError("layout request grid.rows must be an integer")
    if not isinstance(request_columns, int) or isinstance(request_columns, bool):
        raise ValueError("layout request grid.columns must be an integer")
    if not isinstance(request_transport, str):
        raise ValueError("layout request transport must be a string")
    if request_name is not None and not isinstance(request_name, str):
        raise ValueError("layout request name must be a string")
    if not isinstance(terminals, dict):
        raise ValueError("layout request terminals must be an object")
    if not isinstance(expansion, dict):
        raise ValueError("layout request searchExpansion must be an object")
    if not isinstance(constraints, dict):
        raise ValueError("layout request constraints must be an object")
    request_belt_cell = constraints.get("requireSplitterMergerBeltCell", False)
    if not isinstance(request_belt_cell, bool):
        raise ValueError(
            "layout request constraints.requireSplitterMergerBeltCell must be a boolean"
        )
    request_edge_minimums = constraints.get("edgeMinTransitCells", {})
    if not isinstance(request_edge_minimums, dict):
        raise ValueError("layout request constraints.edgeMinTransitCells must be an object")
    for edge_id, minimum in request_edge_minimums.items():
        if not isinstance(edge_id, str):
            raise ValueError("edgeMinTransitCells keys must be edge IDs")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum not in (0, 1, 2):
            raise ValueError(f"edgeMinTransitCells[{edge_id!r}] must be 0, 1, or 2")
    request_auto_expand = expansion.get("enabled", False)
    request_expand_step = expansion.get("step", 1)
    request_max_rows = expansion.get("maxRows", request_rows)
    request_max_columns = expansion.get("maxColumns", request_columns)
    if not isinstance(request_auto_expand, bool):
        raise ValueError("layout request searchExpansion.enabled must be a boolean")
    if not isinstance(request_expand_step, int) or isinstance(request_expand_step, bool):
        raise ValueError("layout request searchExpansion.step must be an integer")
    if not isinstance(request_max_rows, int) or isinstance(request_max_rows, bool):
        raise ValueError("layout request searchExpansion.maxRows must be an integer")
    if not isinstance(request_max_columns, int) or isinstance(request_max_columns, bool):
        raise ValueError("layout request searchExpansion.maxColumns must be an integer")

    return adapt_topoflow(
        graph_input,
        rank=rank if rank is not None else request_rank,
        rows=rows if rows is not None else request_rows,
        columns=columns if columns is not None else request_columns,
        transport=transport if transport is not None else request_transport,
        require_splitter_merger_belt_cell=(
            require_splitter_merger_belt_cell
            if require_splitter_merger_belt_cell is not None
            else request_belt_cell
        ),
        edge_min_transit_cells=(
            edge_min_transit_cells
            if edge_min_transit_cells is not None
            else request_edge_minimums
        ),
        name=name if name is not None else request_name,
        terminal_anchors=terminal_anchors if terminal_anchors is not None else terminals,
        auto_expand=auto_expand if auto_expand is not None else request_auto_expand,
        expand_step=expand_step if expand_step is not None else request_expand_step,
        max_rows=max_rows if max_rows is not None else request_max_rows,
        max_columns=max_columns if max_columns is not None else request_max_columns,
    )


def _port_assignment(
    adapted: AdaptedTopoFlowProblem,
    solution: dict[str, object],
    facility_id: int,
    port_kind: int,
    candidate_index: int,
) -> dict[str, object]:
    machs = solution.get("machs")
    if not isinstance(machs, list) or facility_id >= len(machs) or not isinstance(machs[facility_id], dict):
        raise ValueError("solver solution contains an invalid machs list")
    machine = machs[facility_id]
    position = machine.get("pos")
    rotation = machine.get("rot")
    if not isinstance(position, list) or not isinstance(rotation, int):
        raise ValueError("solver solution is missing facility position or rotation")
    original = adapted.problem.facilities[facility_id].ports[port_kind][candidate_index]
    return {
        "index": candidate_index,
        "kind": port_kind,
        "orig": list(original),
        "cell": list(position),
        "dir": (original[0] + rotation) % 4,
    }


def external_connection_transit_cells(
    connection: Mapping[str, object],
    rows: int,
    columns: int,
) -> int:
    """Return the actual intermediate-cell count from an I/O anchor to its port."""
    if rows < 1 or columns < 1:
        raise ValueError("solution grid dimensions must be positive")
    anchor = connection.get("anchor")
    assignment = connection.get("port_assignment")
    if not isinstance(anchor, Mapping) or not isinstance(assignment, Mapping):
        raise ValueError("external connection is missing its anchor or port assignment")
    side = anchor.get("side")
    offset = anchor.get("offset")
    cell = assignment.get("cell")
    if not isinstance(side, str) or side not in TERMINAL_DIRECTIONS:
        raise ValueError("external connection anchor has an invalid side")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("external connection anchor has an invalid offset")
    if (
        not isinstance(cell, list)
        or len(cell) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
    ):
        raise ValueError("external connection port assignment has an invalid cell")

    if side == "left":
        anchor_cell = (offset, 0)
    elif side == "right":
        anchor_cell = (offset, columns - 1)
    elif side == "top":
        anchor_cell = (0, offset)
    else:
        anchor_cell = (rows - 1, offset)
    return abs(cell[0] - anchor_cell[0]) + abs(cell[1] - anchor_cell[1])


def enrich_solution(
    adapted: AdaptedTopoFlowProblem,
    solution: dict[str, object],
    external_port_indexes: dict[str, int] | None = None,
) -> dict[str, object]:
    machs = solution.get("machs")
    belts = solution.get("belts")
    if not isinstance(machs, list) or not isinstance(belts, list):
        raise ValueError("solver solution must contain machs and belts lists")
    if len(belts) != len(adapted.internal_edges):
        raise ValueError("solver solution connection count no longer matches the TopoFlow graph")

    facility_to_node = {facility: node for node, facility in adapted.node_to_facility.items()}
    for facility_id, machine in enumerate(machs):
        if not isinstance(machine, dict):
            raise ValueError("solver solution mach entry must be an object")
        node = facility_to_node[facility_id]
        machine["topoflow_node_id"] = node
        machine["topoflow_node_type"] = adapted.node_families[node]

    used_ports: dict[tuple[int, int], set[int]] = {}
    for belt, edge in zip(belts, adapted.internal_edges):
        if not isinstance(belt, dict):
            raise ValueError("solver solution belt entry must be an object")
        belt["id"] = edge.edge_id
        belt["topoflow_source"] = edge.source
        belt["topoflow_target"] = edge.target
        assignments = belt.get("port_assignment")
        if not isinstance(assignments, dict):
            raise ValueError("solver solution belt is missing port_assignment")
        for endpoint, facility_key in (("frm", "frm"), ("to", "to")):
            assignment = assignments.get(endpoint)
            port_facility_id = belt.get(facility_key)
            if not isinstance(assignment, dict) or not isinstance(port_facility_id, int):
                raise ValueError("solver solution contains an invalid port assignment")
            port_kind = assignment.get("kind")
            index = assignment.get("index")
            if not isinstance(port_kind, int) or not isinstance(index, int):
                raise ValueError("solver solution port assignment has invalid kind or index")
            used_ports.setdefault((port_facility_id, port_kind), set()).add(index)

    kind = TRANSPORT_KINDS[adapted.transport]
    external_connections: list[dict[str, object]] = []
    for edge in adapted.external_edges:
        if edge.source == "In":
            node = edge.target
            role = "input"
            port_kind = kind
        else:
            node = edge.source
            role = "output"
            port_kind = -kind
        facility_id = adapted.node_to_facility[node]
        candidates = adapted.problem.facilities[facility_id].ports[port_kind]
        used = used_ports.setdefault((facility_id, port_kind), set())
        candidate_index = (
            external_port_indexes.get(edge.edge_id)
            if external_port_indexes is not None
            else next((index for index in range(len(candidates)) if index not in used), None)
        )
        if candidate_index is None:
            raise ValueError(f"no unused physical port remains for external edge {edge.edge_id}")
        if not 0 <= candidate_index < len(candidates) or candidate_index in used:
            raise ValueError(f"invalid or reused physical port for external edge {edge.edge_id}")
        used.add(candidate_index)
        terminal = "In" if edge.source == "In" else "Out"
        anchor = adapted.terminal_anchors.get(terminal)
        connection: dict[str, object] = {
            "id": edge.edge_id,
            "source": edge.source,
            "target": edge.target,
            "transport": adapted.transport,
            "facility_id": facility_id,
            "topoflow_node_id": node,
            "role": role,
            "anchor": (
                {"side": anchor.side, "offset": anchor.offset}
                if anchor is not None
                else None
            ),
            "port_assignment": _port_assignment(
                adapted, solution, facility_id, port_kind, candidate_index
            ),
        }
        if anchor is not None:
            connection["transit_cells"] = external_connection_transit_cells(
                connection,
                adapted.problem.rows,
                adapted.problem.columns,
            )
        external_connections.append(connection)

    solution["topoflow"] = {
        "schema": SOLUTION_SCHEMA,
        "selection": adapted.selection,
        "transport": adapted.transport,
        "constraints": {
            "requireSplitterMergerBeltCell": adapted.require_splitter_merger_belt_cell,
            "edgeMinTransitCells": adapted.edge_min_transit_cells,
        },
        "graph": adapted.graph,
        "node_to_facility": adapted.node_to_facility,
        "terminals": {
            terminal: {"side": anchor.side, "offset": anchor.offset}
            for terminal, anchor in adapted.terminal_anchors.items()
        },
        "external_connections": external_connections,
    }
    return solution


def _add_terminal_constraints(
    adapted: AdaptedTopoFlowProblem,
    artifacts: solver_module.ModelArtifacts,
) -> dict[str, Any]:
    if not adapted.terminal_anchors:
        return {}

    kind = TRANSPORT_KINDS[adapted.transport]
    choices: dict[str, object] = {}
    used_by_port: dict[tuple[int, int], list[object]] = {}

    for edge, variables in zip(adapted.internal_edges, artifacts.connections):
        source_id = adapted.node_to_facility[edge.source]
        target_id = adapted.node_to_facility[edge.target]
        used_by_port.setdefault((source_id, -kind), []).append(variables.output_choice)
        used_by_port.setdefault((target_id, kind), []).append(variables.input_choice)

    for edge in adapted.external_edges:
        if edge.source == "In":
            terminal = "In"
            node = edge.target
            port_kind = kind
        else:
            terminal = "Out"
            node = edge.source
            port_kind = -kind
        anchor = adapted.terminal_anchors[terminal]
        facility_id = adapted.node_to_facility[node]
        facility_vars = artifacts.facilities[facility_id]
        ports = adapted.problem.facilities[facility_id].ports[port_kind]
        choice = artifacts.model.new_int_var(
            0, len(ports) - 1, f"{edge.edge_id.replace('-', '_')}_external_port"
        )
        required_direction = TERMINAL_DIRECTIONS[anchor.side]
        allowed = [
            (candidate, rotation)
            for candidate, port in enumerate(ports)
            for rotation in range(4)
            if (port[0] + rotation) % 4 == required_direction
        ]
        artifacts.model.add_allowed_assignments([choice, facility_vars.rotation], allowed)

        if anchor.side == "left":
            artifacts.model.add(facility_vars.row == anchor.offset)
            artifacts.model.add(facility_vars.column == 0)
        elif anchor.side == "right":
            artifacts.model.add(facility_vars.row == anchor.offset)
            artifacts.model.add(facility_vars.column == adapted.problem.columns - 1)
        elif anchor.side == "top":
            artifacts.model.add(facility_vars.row == 0)
            artifacts.model.add(facility_vars.column == anchor.offset)
        else:
            artifacts.model.add(facility_vars.row == adapted.problem.rows - 1)
            artifacts.model.add(facility_vars.column == anchor.offset)

        port_key = (facility_id, port_kind)
        for used_choice in used_by_port.get(port_key, []):
            artifacts.model.add(choice != used_choice)
        used_by_port.setdefault(port_key, []).append(choice)
        choices[edge.edge_id] = choice
    return choices


def _terminal_positions(adapted: AdaptedTopoFlowProblem) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    for edge in adapted.external_edges:
        terminal = "In" if edge.source == "In" else "Out"
        anchor = adapted.terminal_anchors.get(terminal)
        if anchor is None:
            continue
        node = edge.target if terminal == "In" else edge.source
        facility_id = adapted.node_to_facility[node]
        if anchor.side == "left":
            position = (anchor.offset, 0)
        elif anchor.side == "right":
            position = (anchor.offset, adapted.problem.columns - 1)
        elif anchor.side == "top":
            position = (0, anchor.offset)
        else:
            position = (adapted.problem.rows - 1, anchor.offset)
        previous = positions.get(facility_id)
        if previous is not None and previous != position:
            raise ValueError(
                "In and Out anchors constrain the same facility to different cells"
            )
        positions[facility_id] = position
    return positions


def normalize_layout_hint(
    adapted: AdaptedTopoFlowProblem,
    layout_hint: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(layout_hint, Mapping):
        return None
    raw_facilities = layout_hint.get("facilities")
    if not isinstance(raw_facilities, list):
        return None
    anchors = _terminal_positions(adapted)
    positions: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for item in raw_facilities:
        if not isinstance(item, Mapping):
            continue
        facility_id = item.get("id")
        row = item.get("row")
        column = item.get("column")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (facility_id, row, column)
        ):
            continue
        assert isinstance(facility_id, int) and isinstance(row, int) and isinstance(column, int)
        position = (row, column)
        if not 0 <= facility_id < len(adapted.problem.facilities):
            continue
        if facility_id in positions:
            continue
        if not (0 <= row < adapted.problem.rows and 0 <= column < adapted.problem.columns):
            continue
        if facility_id in anchors and anchors[facility_id] != position:
            continue
        if position in occupied:
            continue
        positions[facility_id] = position
        occupied.add(position)
    if not positions:
        return None
    return {
        "schema": LAYOUT_HINT_SCHEMA,
        "grid": {"rows": adapted.problem.rows, "columns": adapted.problem.columns},
        "facilities": [
            {"id": facility_id, "row": row, "column": column}
            for facility_id, (row, column) in sorted(positions.items())
        ],
        "source": str(layout_hint.get("source") or "reused"),
    }


def solve_facility_placement(
    adapted: AdaptedTopoFlowProblem,
    cp_model: Any,
    *,
    time_limit: float,
    workers: int,
    prior_hint: Mapping[str, object] | None = None,
    debug: bool = False,
) -> dict[str, object] | None:
    """Quickly place facilities without grid-routing variables.

    The result is only a CP-SAT hint for the complete model.  It never becomes
    a hard placement constraint, so a later router can freely improve or move
    every non-terminal facility.
    """
    if time_limit <= 0:
        return normalize_layout_hint(adapted, prior_hint)
    model = cp_model.CpModel()
    rows: list[Any] = []
    columns: list[Any] = []
    cells: list[Any] = []
    cell_count = adapted.problem.rows * adapted.problem.columns
    for facility_id in range(len(adapted.problem.facilities)):
        row = model.new_int_var(0, adapted.problem.rows - 1, f"placement_{facility_id}_row")
        column = model.new_int_var(0, adapted.problem.columns - 1, f"placement_{facility_id}_column")
        cell = model.new_int_var(0, cell_count - 1, f"placement_{facility_id}_cell")
        model.add(cell == row * adapted.problem.columns + column)
        rows.append(row)
        columns.append(column)
        cells.append(cell)
    model.add_all_different(cells)

    for facility_id, (row_value, column_value) in _terminal_positions(adapted).items():
        model.add(rows[facility_id] == row_value)
        model.add(columns[facility_id] == column_value)

    distances: list[Any] = []
    for edge in adapted.internal_edges:
        source = adapted.node_to_facility[edge.source]
        target = adapted.node_to_facility[edge.target]
        row_delta = model.new_int_var(0, adapted.problem.rows - 1, f"placement_{edge.edge_id}_dr")
        column_delta = model.new_int_var(0, adapted.problem.columns - 1, f"placement_{edge.edge_id}_dc")
        distance = model.new_int_var(
            1,
            max(1, adapted.problem.rows + adapted.problem.columns - 2),
            f"placement_{edge.edge_id}_distance",
        )
        model.add_abs_equality(row_delta, rows[source] - rows[target])
        model.add_abs_equality(column_delta, columns[source] - columns[target])
        model.add(distance == row_delta + column_delta)
        distances.append(distance)
    if distances:
        model.minimize(sum(distances))

    normalized = normalize_layout_hint(adapted, prior_hint)
    if normalized is not None:
        facilities = normalized["facilities"]
        assert isinstance(facilities, list)
        for item in facilities:
            assert isinstance(item, dict)
            facility_id = int(item["id"])
            row_value = int(item["row"])
            column_value = int(item["column"])
            model.add_hint(rows[facility_id], row_value)
            model.add_hint(columns[facility_id], column_value)
            model.add_hint(cells[facility_id], row_value * adapted.problem.columns + column_value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.relative_gap_limit = 0.15
    solver.parameters.log_search_progress = debug
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return normalized
    return {
        "schema": LAYOUT_HINT_SCHEMA,
        "grid": {"rows": adapted.problem.rows, "columns": adapted.problem.columns},
        "facilities": [
            {
                "id": facility_id,
                "row": solver.value(rows[facility_id]),
                "column": solver.value(columns[facility_id]),
            }
            for facility_id in range(len(rows))
        ],
        "source": "placement-pre-solve",
        "objective": int(solver.objective_value) if distances else 0,
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
    }


def _configure_search_mode(solver: Any, search_mode: str) -> None:
    if search_mode not in SEARCH_MODES:
        raise ValueError(f"search_mode must be one of {sorted(SEARCH_MODES)}")
    if search_mode == "fast":
        solver.parameters.stop_after_first_solution = True
    elif search_mode == "balanced":
        solver.parameters.relative_gap_limit = 0.05


def solve_topoflow(
    adapted: AdaptedTopoFlowProblem,
    *,
    time_limit: float = 300.0,
    workers: int = 8,
    debug: bool = False,
    search_mode: str = "balanced",
    wall_time_limit: float | None = None,
    layout_hint: Mapping[str, object] | None = None,
    route_length_upper_bound: int | None = None,
    layout_hint_callback: LayoutHintCallback | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], str, int]:
    if time_limit <= 0 or workers <= 0:
        raise ValueError("time_limit and workers must be positive")
    if search_mode not in SEARCH_MODES:
        raise ValueError(f"search_mode must be one of {sorted(SEARCH_MODES)}")
    if wall_time_limit is not None and wall_time_limit <= 0:
        raise ValueError("wall_time_limit must be positive")
    if (
        route_length_upper_bound is not None
        and (
            not isinstance(route_length_upper_bound, int)
            or isinstance(route_length_upper_bound, bool)
            or route_length_upper_bound < 0
        )
    ):
        raise ValueError("route_length_upper_bound must be a non-negative integer")
    started_at = time.monotonic()
    import ilpbridge as cp_model

    available_for_placement = wall_time_limit
    if available_for_placement is None:
        available_for_placement = time_limit
    placement_budget = min(
        5.0,
        max(0.5, available_for_placement * 0.1),
        max(0.05, available_for_placement - 0.05),
    )
    if progress_callback is not None:
        progress_callback("placement")
    effective_hint = solve_facility_placement(
        adapted,
        cp_model,
        time_limit=placement_budget,
        workers=workers,
        prior_hint=layout_hint,
        debug=debug,
    )
    if effective_hint is not None and layout_hint_callback is not None:
        layout_hint_callback(effective_hint)

    if progress_callback is not None:
        progress_callback("building")
    artifacts = solver_module.build_model(adapted.problem, cp_model, effective_hint)
    if route_length_upper_bound is not None:
        artifacts.model.add(
            sum(variable for connection in artifacts.connections for variable in connection.arcs)
            <= route_length_upper_bound
        )
    terminal_choices = _add_terminal_constraints(adapted, artifacts)
    effective_time_limit = time_limit
    if wall_time_limit is not None:
        remaining = wall_time_limit - (time.monotonic() - started_at)
        if remaining <= 0.05:
            raise SolveFailure("TL (model build exhausted attempt budget)", retryable=True)
        effective_time_limit = min(effective_time_limit, remaining)
    if progress_callback is not None:
        progress_callback("solving")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = effective_time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.log_search_progress = debug
    _configure_search_mode(solver, search_mode)
    status = solver.solve(artifacts.model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        label = {
            cp_model.INFEASIBLE: "NO (infeasible)",
            cp_model.MODEL_INVALID: "Invalid model",
            cp_model.UNKNOWN: "TL (no solution before timeout)",
        }.get(status, f"Solver status {status}")
        raise SolveFailure(
            label,
            retryable=status in (cp_model.INFEASIBLE, cp_model.UNKNOWN),
        )

    solution = solver_module.build_solution(adapted.problem, artifacts, solver)
    external_port_indexes = {
        edge_id: solver.value(choice)
        for edge_id, choice in terminal_choices.items()
    }
    enrich_solution(
        adapted,
        solution,
        external_port_indexes if terminal_choices else None,
    )
    label = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    return solution, label, int(solver.objective_value)


def _expanded_grid_sizes(adapted: AdaptedTopoFlowProblem) -> list[tuple[int, int]]:
    expansion = adapted.search_expansion
    sizes = [(adapted.problem.rows, adapted.problem.columns)]
    if not expansion.enabled:
        return sizes
    rows, columns = sizes[0]
    while rows < expansion.max_rows or columns < expansion.max_columns:
        next_rows = min(rows + expansion.step, expansion.max_rows)
        next_columns = min(columns + expansion.step, expansion.max_columns)
        if (next_rows, next_columns) == (rows, columns):
            break
        sizes.append((next_rows, next_columns))
        rows, columns = next_rows, next_columns
    return sizes


def _resize_adapted_problem(
    adapted: AdaptedTopoFlowProblem,
    rows: int,
    columns: int,
) -> AdaptedTopoFlowProblem:
    selected: dict[str, object] = dict(adapted.selection)
    selected["graph"] = adapted.graph
    anchors: dict[str, object] = {
        terminal: {"side": anchor.side, "offset": anchor.offset}
        for terminal, anchor in adapted.terminal_anchors.items()
    }
    rank_value = adapted.selection.get("rank", 1)
    rank = rank_value if isinstance(rank_value, int) else 1
    return adapt_topoflow(
        selected,
        rank=rank,
        rows=rows,
        columns=columns,
        transport=adapted.transport,
        require_splitter_merger_belt_cell=adapted.require_splitter_merger_belt_cell,
        edge_min_transit_cells=adapted.edge_min_transit_cells,
        name=adapted.problem.metadata.get("name") or adapted.problem.fallback_name,
        terminal_anchors=anchors or None,
        auto_expand=adapted.search_expansion.enabled,
        expand_step=adapted.search_expansion.step,
        max_rows=adapted.search_expansion.max_rows,
        max_columns=adapted.search_expansion.max_columns,
    )


def _original_design_metadata(adapted: AdaptedTopoFlowProblem) -> dict[str, object]:
    selected = dict(adapted.selection)
    selected["graph"] = adapted.graph
    rank_value = adapted.selection.get("rank", 1)
    rank = rank_value if isinstance(rank_value, int) else 1
    expansion = adapted.search_expansion
    return {
        "schema": LAYOUT_REQUEST_SCHEMA,
        "name": adapted.problem.metadata.get("name") or adapted.problem.fallback_name,
        "rank": rank,
        "grid": {
            "rows": adapted.problem.rows,
            "columns": adapted.problem.columns,
        },
        "transport": adapted.transport,
        "constraints": {
            "requireSplitterMergerBeltCell": adapted.require_splitter_merger_belt_cell,
            "edgeMinTransitCells": adapted.edge_min_transit_cells,
        },
        "terminals": {
            terminal: {"side": anchor.side, "offset": anchor.offset}
            for terminal, anchor in adapted.terminal_anchors.items()
        },
        "searchExpansion": {
            "enabled": expansion.enabled,
            "step": expansion.step,
            "maxRows": expansion.max_rows,
            "maxColumns": expansion.max_columns,
        },
        "topoflow": selected,
    }


def solve_with_expansion(
    adapted: AdaptedTopoFlowProblem,
    *,
    time_limit: float = 300.0,
    workers: int = 8,
    debug: bool = False,
    search_mode: str = "balanced",
    wall_time_limit: float | None = None,
    layout_hint: Mapping[str, object] | None = None,
    route_length_upper_bound: int | None = None,
    layout_hint_callback: LayoutHintCallback | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, object], str, int, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    sizes = _expanded_grid_sizes(adapted)
    last_failure: SolveFailure | None = None
    started_at = time.monotonic()
    current_hint = normalize_layout_hint(adapted, layout_hint)
    for rows, columns in sizes:
        current = (
            adapted
            if (rows, columns) == (adapted.problem.rows, adapted.problem.columns)
            else _resize_adapted_problem(adapted, rows, columns)
        )
        try:
            remaining_wall_time = None
            if wall_time_limit is not None:
                remaining_wall_time = wall_time_limit - (time.monotonic() - started_at)
                if remaining_wall_time <= 0.05:
                    raise SolveFailure("TL (physical stage budget exhausted)", retryable=True)
            callback = (
                (lambda phase, attempt_rows=rows, attempt_columns=columns: progress_callback(
                    phase, attempt_rows, attempt_columns
                ))
                if progress_callback is not None
                else None
            )

            def capture_layout_hint(value: dict[str, object]) -> None:
                nonlocal current_hint
                current_hint = value
                if layout_hint_callback is not None:
                    layout_hint_callback(value)

            solution, label, route_length = solve_topoflow(
                current,
                time_limit=time_limit,
                workers=workers,
                debug=debug,
                search_mode=search_mode,
                wall_time_limit=remaining_wall_time,
                layout_hint=current_hint,
                route_length_upper_bound=route_length_upper_bound,
                layout_hint_callback=capture_layout_hint,
                progress_callback=callback,
            )
            attempts.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "status": label,
                    "routeLength": route_length,
                }
            )
            topoflow = solution.get("topoflow")
            if not isinstance(topoflow, dict):
                raise ValueError("solver solution is missing topoflow metadata")
            expansion = adapted.search_expansion
            topoflow["grid_search"] = {
                "enabled": expansion.enabled,
                "initialGrid": {
                    "rows": adapted.problem.rows,
                    "columns": adapted.problem.columns,
                },
                "finalGrid": {"rows": rows, "columns": columns},
                "step": expansion.step,
                "maxRows": expansion.max_rows,
                "maxColumns": expansion.max_columns,
                "attempts": attempts,
            }
            topoflow["original_design"] = _original_design_metadata(adapted)
            return solution, label, route_length, attempts
        except SolveFailure as failure:
            attempts.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "status": failure.label,
                }
            )
            last_failure = failure
            if not failure.retryable:
                break

    summary = ", ".join(
        f"{attempt['rows']}x{attempt['columns']}={attempt['status']}"
        for attempt in attempts
    )
    if last_failure is None:
        raise SolveFailure("no grid sizes were attempted", retryable=False)
    raise SolveFailure(
        f"{last_failure.label}; grid attempts: {summary}",
        retryable=last_failure.retryable,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place and route a selected TopoFlow GA topology on an Endfield grid"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="visual .layout.json, TopoFlow ga_top5.json, result object, or bare graph JSON",
    )
    parser.add_argument("--rank", type=int, help="override TopoFlow result rank")
    parser.add_argument("--rows", type=int, help="override layout grid rows")
    parser.add_argument("--columns", type=int, help="override layout grid columns")
    parser.add_argument("--transport", choices=tuple(TRANSPORT_KINDS), help="override transport")
    parser.add_argument(
        "--require-splitter-merger-belt-cell",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="require at least one ordinary belt cell from a splitter to a merger",
    )
    parser.add_argument(
        "--auto-expand",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="retry on progressively larger grids",
    )
    parser.add_argument("--expand-step", type=int, help="rows/columns added per retry")
    parser.add_argument("--max-rows", type=int, help="maximum rows for automatic expansion")
    parser.add_argument("--max-columns", type=int, help="maximum columns for automatic expansion")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."))
    parser.add_argument("--name", help="override the solution filename prefix")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--search-mode",
        choices=tuple(sorted(SEARCH_MODES)),
        default="balanced",
        help="CP-SAT search policy (default: balanced)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        fallback_name = args.input.name.removesuffix(".json")
        if isinstance(raw, dict) and raw.get("schema") == LAYOUT_REQUEST_SCHEMA:
            adapted = adapt_layout_request(
                raw,
                rank=args.rank,
                rows=args.rows,
                columns=args.columns,
                transport=args.transport,
                require_splitter_merger_belt_cell=args.require_splitter_merger_belt_cell,
                name=args.name,
                auto_expand=args.auto_expand,
                expand_step=args.expand_step,
                max_rows=args.max_rows,
                max_columns=args.max_columns,
            )
        else:
            selected_rank = args.rank if args.rank is not None else 1
            adapted = adapt_topoflow(
                raw,
                rank=selected_rank,
                rows=args.rows if args.rows is not None else 12,
                columns=args.columns if args.columns is not None else 12,
                transport=args.transport if args.transport is not None else "belt",
                require_splitter_merger_belt_cell=(
                    args.require_splitter_merger_belt_cell
                    if args.require_splitter_merger_belt_cell is not None
                    else False
                ),
                name=args.name or f"{fallback_name}_rank_{selected_rank}",
                auto_expand=args.auto_expand if args.auto_expand is not None else False,
                expand_step=args.expand_step if args.expand_step is not None else 1,
                max_rows=args.max_rows,
                max_columns=args.max_columns,
            )
        solution, label, route_length, attempts = solve_with_expansion(
            adapted,
            time_limit=args.time_limit,
            workers=args.workers,
            debug=args.debug,
            search_mode=args.search_mode,
        )
        output_path = solver_module.resolve_output_path(
            args.output_dir, args.name, adapted.problem
        )
        output_path.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, (OSError, json.JSONDecodeError, ValueError)) else 1

    print(f"OK ({label}), route length = {route_length}")
    if len(attempts) > 1:
        print("Grid search: " + " -> ".join(
            f"{attempt['rows']}x{attempt['columns']} ({attempt['status']})"
            for attempt in attempts
        ))
    print(f"Solution exported to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
