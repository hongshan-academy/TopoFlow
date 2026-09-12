#!/usr/bin/env python3
"""Minimal CP-SAT solver for Endfield splitters and mergers only.

Accepted facilities are 1x1 ``logi`` nodes.  The program keeps the original
``mach`` / ``belt`` input schema and writes an editor-compatible solution JSON,
but deliberately omits machines, power, storage, and large-facility geometry.
Multi-port splitter outputs and merger inputs are interchangeable decisions:
the solver assigns each logical connection to the physical port that best fits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# side 0/1/2/3 = down/right/up/left, matching the root solver and editor.
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
PORT_KINDS = (-2, -1, 1, 2)


DEFAULT_PROBLEM: dict[str, object] = {
    "name": "logistics_demo",
    "description": "One belt splitter feeding three mergers, then one merger.",
    "n": 5,
    "m": 5,
    "mach": [
        {
            "type": "logi",
            "size": [1, 1],
            "1": [[2, 0]],
            "-1": [[0, 0], [1, 0], [3, 0]],
        },
        *[
            {
                "type": "logi",
                "size": [1, 1],
                "1": [[0, 0], [1, 0], [3, 0]],
                "-1": [[2, 0]],
            }
            for _ in range(4)
        ],
    ],
    "belt": [
        [1, 0, 1],
        [1, 0, 2],
        [1, 0, 3],
        [1, 1, 4],
        [1, 2, 4],
        [1, 3, 4],
    ],
}


@dataclass(frozen=True)
class Facility:
    ports: dict[int, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class Connection:
    kind: int
    source: int
    target: int


@dataclass(frozen=True)
class Problem:
    rows: int
    columns: int
    facilities: tuple[Facility, ...]
    connections: tuple[Connection, ...]
    require_splitter_merger_belt_cell: bool
    edge_min_transit_cells: tuple[int, ...]
    metadata: dict[str, str]
    raw: dict[str, object]
    fallback_name: str


@dataclass
class FacilityVariables:
    row: Any
    column: Any
    cell: Any
    rotation: Any
    at_cell: list[Any]
    rotation_is: list[Any]


@dataclass
class ConnectionVariables:
    arcs: list[Any]
    transit: list[Any]
    output_choice: Any
    input_choice: Any


@dataclass
class ModelArtifacts:
    model: Any
    facilities: list[FacilityVariables]
    connections: list[ConnectionVariables]
    directed_arcs: list[tuple[int, int]]
    undirected_edges: list[tuple[int, int]]


def _as_int(value: object, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def _parse_ports(raw_facility: dict[str, object], facility_id: int) -> dict[int, tuple[tuple[int, int], ...]]:
    ports: dict[int, tuple[tuple[int, int], ...]] = {}
    occupied_sides: set[int] = set()
    for kind in PORT_KINDS:
        raw_ports = raw_facility.get(str(kind), [])
        if not isinstance(raw_ports, list):
            raise ValueError(f"mach[{facility_id}][{kind}] must be a list")
        parsed: list[tuple[int, int]] = []
        for port_id, raw_port in enumerate(raw_ports):
            if not isinstance(raw_port, list) or len(raw_port) != 2:
                raise ValueError(f"mach[{facility_id}][{kind}][{port_id}] must be [side, 0]")
            side = _as_int(raw_port[0], f"mach[{facility_id}][{kind}][{port_id}].side")
            offset = _as_int(raw_port[1], f"mach[{facility_id}][{kind}][{port_id}].offset")
            if side not in range(4) or offset != 0:
                raise ValueError(f"mach[{facility_id}] is 1x1, so every port must be [side 0..3, 0]")
            if side in occupied_sides:
                raise ValueError(f"mach[{facility_id}] declares more than one port on side {side}")
            occupied_sides.add(side)
            parsed.append((side, offset))
        ports[kind] = tuple(parsed)
    return ports


def parse_problem(raw: dict[str, object], fallback_name: str = "logistics") -> Problem:
    for key in ("mach", "belt", "n", "m"):
        if key not in raw:
            raise ValueError(f"missing required field: {key}")

    rows = _as_int(raw["n"], "n")
    columns = _as_int(raw["m"], "m")
    if rows < 1 or columns < 1:
        raise ValueError("n and m must be positive")

    raw_facilities = raw["mach"]
    if not isinstance(raw_facilities, list) or not raw_facilities:
        raise ValueError("mach must be a non-empty list")
    facilities: list[Facility] = []
    for facility_id, item in enumerate(raw_facilities):
        if not isinstance(item, dict):
            raise ValueError(f"mach[{facility_id}] must be an object")
        if item.get("type") != "logi" or item.get("size") != [1, 1]:
            raise ValueError(f"mach[{facility_id}] must have type='logi' and size=[1, 1]")
        ports = _parse_ports(item, facility_id)
        shapes = [
            (len(ports[-kind]), len(ports[kind]))
            for kind in (1, 2)
            if ports[-kind] or ports[kind]
        ]
        if len(shapes) != 1 or shapes[0] not in ((3, 1), (1, 3)):
            raise ValueError(
                f"mach[{facility_id}] must be one belt/pipe splitter (1 in, 3 out) "
                "or merger (3 in, 1 out)"
            )
        facilities.append(Facility(ports=ports))

    if len(facilities) > rows * columns:
        raise ValueError("the grid has fewer cells than facilities")

    raw_connections = raw["belt"]
    if not isinstance(raw_connections, list):
        raise ValueError("belt must be a list")
    connections: list[Connection] = []
    endpoint_usage: dict[tuple[int, int], int] = {}
    for connection_id, item in enumerate(raw_connections):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"belt[{connection_id}] must be [type, source, target]")
        kind, source, target = (_as_int(value, f"belt[{connection_id}]") for value in item)
        if kind not in (1, 2):
            raise ValueError(f"belt[{connection_id}] type must be 1 (belt) or 2 (pipe)")
        if not (0 <= source < len(facilities) and 0 <= target < len(facilities)):
            raise ValueError(f"belt[{connection_id}] references an unknown facility")
        if source == target:
            raise ValueError(f"belt[{connection_id}] cannot connect a facility to itself")
        if not facilities[source].ports[-kind] or not facilities[target].ports[kind]:
            raise ValueError(f"belt[{connection_id}] has no compatible source or target port")
        connections.append(Connection(kind=kind, source=source, target=target))
        endpoint_usage[(source, -kind)] = endpoint_usage.get((source, -kind), 0) + 1
        endpoint_usage[(target, kind)] = endpoint_usage.get((target, kind), 0) + 1

    for (facility_id, port_kind), count in endpoint_usage.items():
        available = len(facilities[facility_id].ports[port_kind])
        if count > available:
            raise ValueError(
                f"mach[{facility_id}] needs {count} uses of port kind {port_kind}, "
                f"but only declares {available}"
            )

    raw_constraints = raw.get("constraints", {})
    if not isinstance(raw_constraints, dict):
        raise ValueError("constraints must be an object")
    require_splitter_merger_belt_cell = raw_constraints.get(
        "require_splitter_merger_belt_cell", False
    )
    if not isinstance(require_splitter_merger_belt_cell, bool):
        raise ValueError("constraints.require_splitter_merger_belt_cell must be a boolean")
    raw_edge_minimums = raw_constraints.get("edge_min_transit_cells")
    if raw_edge_minimums is None:
        edge_min_transit_cells = [0] * len(connections)
    else:
        if not isinstance(raw_edge_minimums, list):
            raise ValueError("constraints.edge_min_transit_cells must be a list")
        if len(raw_edge_minimums) != len(connections):
            raise ValueError(
                "constraints.edge_min_transit_cells must match the belt connection count"
            )
        edge_min_transit_cells = []
        for connection_id, value in enumerate(raw_edge_minimums):
            minimum = _as_int(value, f"constraints.edge_min_transit_cells[{connection_id}]")
            if minimum < 0 or minimum > 2:
                raise ValueError("edge minimum transit cells must be 0, 1, or 2")
            edge_min_transit_cells.append(minimum)

    metadata = {
        key: str(raw[key])
        for key in ("name", "description", "note")
        if raw.get(key) not in (None, "")
    }
    return Problem(
        rows=rows,
        columns=columns,
        facilities=tuple(facilities),
        connections=tuple(connections),
        require_splitter_merger_belt_cell=require_splitter_merger_belt_cell,
        edge_min_transit_cells=tuple(edge_min_transit_cells),
        metadata=metadata,
        raw=raw,
        fallback_name=fallback_name,
    )


def _grid_edges(rows: int, columns: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    undirected: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            cell = row * columns + column
            if row + 1 < rows:
                undirected.append((cell, (row + 1) * columns + column))
            if column + 1 < columns:
                undirected.append((cell, cell + 1))
    directed = [arc for u, v in undirected for arc in ((u, v), (v, u))]
    return undirected, directed


def _hinted_positions(
    layout_hint: Mapping[str, object] | None,
    facility_count: int,
    rows: int,
    columns: int,
) -> dict[int, tuple[int, int]]:
    if not isinstance(layout_hint, Mapping):
        return {}
    raw_facilities = layout_hint.get("facilities")
    if not isinstance(raw_facilities, list):
        return {}
    positions: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for item in raw_facilities:
        if not isinstance(item, Mapping):
            continue
        facility_id = item.get("id")
        row = item.get("row")
        column = item.get("column")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (facility_id, row, column)):
            continue
        assert isinstance(facility_id, int) and isinstance(row, int) and isinstance(column, int)
        position = (row, column)
        if not 0 <= facility_id < facility_count:
            continue
        if not (0 <= row < rows and 0 <= column < columns):
            continue
        if position in occupied:
            continue
        positions[facility_id] = position
        occupied.add(position)
    return positions


def build_model(
    problem: Problem,
    cp_model: Any,
    layout_hint: Mapping[str, object] | None = None,
) -> ModelArtifacts:
    model = cp_model.CpModel()
    cell_count = problem.rows * problem.columns
    undirected_edges, directed_arcs = _grid_edges(problem.rows, problem.columns)
    arc_index = {arc: index for index, arc in enumerate(directed_arcs)}
    outgoing: list[list[int]] = [[] for _ in range(cell_count)]
    incoming: list[list[int]] = [[] for _ in range(cell_count)]
    for index, (source, target) in enumerate(directed_arcs):
        outgoing[source].append(index)
        incoming[target].append(index)

    facility_variables: list[FacilityVariables] = []
    for facility_id in range(len(problem.facilities)):
        row = model.NewIntVar(0, problem.rows - 1, f"facility_{facility_id}_row")
        column = model.NewIntVar(0, problem.columns - 1, f"facility_{facility_id}_column")
        cell = model.NewIntVar(0, cell_count - 1, f"facility_{facility_id}_cell")
        rotation = model.NewIntVar(0, 3, f"facility_{facility_id}_rotation")
        model.Add(cell == row * problem.columns + column)

        at_cell = [model.NewBoolVar(f"facility_{facility_id}_at_{node}") for node in range(cell_count)]
        model.AddExactlyOne(at_cell)
        for node, flag in enumerate(at_cell):
            model.Add(cell == node).OnlyEnforceIf(flag)
            model.Add(cell != node).OnlyEnforceIf(flag.Not())

        rotation_is = [model.NewBoolVar(f"facility_{facility_id}_rot_{direction}") for direction in range(4)]
        model.AddExactlyOne(rotation_is)
        for direction, flag in enumerate(rotation_is):
            model.Add(rotation == direction).OnlyEnforceIf(flag)
            model.Add(rotation != direction).OnlyEnforceIf(flag.Not())
        facility_variables.append(FacilityVariables(row, column, cell, rotation, at_cell, rotation_is))

    model.AddAllDifferent([facility.cell for facility in facility_variables])

    for facility_id, (row_value, column_value) in _hinted_positions(
        layout_hint,
        len(facility_variables),
        problem.rows,
        problem.columns,
    ).items():
        variables = facility_variables[facility_id]
        model.AddHint(variables.row, row_value)
        model.AddHint(variables.column, column_value)
        model.AddHint(variables.cell, row_value * problem.columns + column_value)

    connection_variables: list[ConnectionVariables] = []
    # A logical connection does not own a fixed physical port. Each endpoint
    # chooses any compatible candidate, while AllDifferent below prevents reuse.
    port_choices: dict[tuple[int, int], list[Any]] = {}
    for connection_id, connection in enumerate(problem.connections):
        source_ports = problem.facilities[connection.source].ports[-connection.kind]
        target_ports = problem.facilities[connection.target].ports[connection.kind]
        output_choice = model.NewIntVar(0, len(source_ports) - 1, f"connection_{connection_id}_output")
        input_choice = model.NewIntVar(0, len(target_ports) - 1, f"connection_{connection_id}_input")
        output_direction = model.NewIntVar(0, 3, f"connection_{connection_id}_output_direction")
        input_direction = model.NewIntVar(0, 3, f"connection_{connection_id}_input_direction")
        model.AddAllowedAssignments(
            [output_choice, facility_variables[connection.source].rotation, output_direction],
            [(choice, rotation, (port[0] + rotation) % 4) for choice, port in enumerate(source_ports) for rotation in range(4)],
        )
        model.AddAllowedAssignments(
            [input_choice, facility_variables[connection.target].rotation, input_direction],
            [(choice, rotation, (port[0] + rotation) % 4) for choice, port in enumerate(target_ports) for rotation in range(4)],
        )
        port_choices.setdefault((connection.source, -connection.kind), []).append(output_choice)
        port_choices.setdefault((connection.target, connection.kind), []).append(input_choice)

        output_direction_is = [model.NewBoolVar(f"connection_{connection_id}_out_dir_{d}") for d in range(4)]
        input_direction_is = [model.NewBoolVar(f"connection_{connection_id}_in_dir_{d}") for d in range(4)]
        for direction in range(4):
            model.Add(output_direction == direction).OnlyEnforceIf(output_direction_is[direction])
            model.Add(output_direction != direction).OnlyEnforceIf(output_direction_is[direction].Not())
            model.Add(input_direction == direction).OnlyEnforceIf(input_direction_is[direction])
            model.Add(input_direction != direction).OnlyEnforceIf(input_direction_is[direction].Not())

        arcs = [model.NewBoolVar(f"connection_{connection_id}_arc_{index}") for index in range(len(directed_arcs))]
        order = [model.NewIntVar(0, cell_count - 1, f"connection_{connection_id}_order_{node}") for node in range(cell_count)]
        transit = [model.NewBoolVar(f"connection_{connection_id}_transit_{node}") for node in range(cell_count)]
        source_vars = facility_variables[connection.source]
        target_vars = facility_variables[connection.target]
        for node in range(cell_count):
            in_degree = sum(arcs[index] for index in incoming[node])
            out_degree = sum(arcs[index] for index in outgoing[node])
            model.Add(out_degree - in_degree == source_vars.at_cell[node] - target_vars.at_cell[node])
            model.Add(in_degree <= 1 - source_vars.at_cell[node])
            model.Add(out_degree <= 1 - target_vars.at_cell[node])
            model.Add(transit[node] <= in_degree)
            model.Add(transit[node] <= out_degree)
            model.Add(transit[node] >= in_degree + out_degree - 1)
            model.Add(order[node] == 0).OnlyEnforceIf(source_vars.at_cell[node])

            row, column = divmod(node, problem.columns)
            for direction, (dr, dc) in enumerate(DIRECTIONS):
                next_row, next_column = row + dr, column + dc
                if 0 <= next_row < problem.rows and 0 <= next_column < problem.columns:
                    neighbor = next_row * problem.columns + next_column
                    model.Add(arcs[arc_index[(node, neighbor)]] == 1).OnlyEnforceIf(
                        [source_vars.at_cell[node], output_direction_is[direction]]
                    )
                    model.Add(arcs[arc_index[(neighbor, node)]] == 1).OnlyEnforceIf(
                        [target_vars.at_cell[node], input_direction_is[direction]]
                    )
                else:
                    model.AddBoolOr([source_vars.at_cell[node].Not(), output_direction_is[direction].Not()])
                    model.AddBoolOr([target_vars.at_cell[node].Not(), input_direction_is[direction].Not()])

        for arc_id, (source, target) in enumerate(directed_arcs):
            model.Add(order[target] >= order[source] + 1).OnlyEnforceIf(arcs[arc_id])

        # A route may start/end on its own facilities, but it cannot use any
        # occupied grid cell as an intermediate transit cell.  Aggregating by
        # cell is equivalent to forbidding every incident arc facility-by-
        # facility, while avoiding millions of repeated enforced constraints.
        for node in range(cell_count):
            model.Add(
                transit[node]
                + sum(facility.at_cell[node] for facility in facility_variables)
                <= 1
            )

        if problem.require_splitter_merger_belt_cell and connection.kind == 1:
            source_facility = problem.facilities[connection.source]
            target_facility = problem.facilities[connection.target]
            source_is_splitter = (
                len(source_facility.ports[1]) == 1
                and len(source_facility.ports[-1]) == 3
            )
            target_is_merger = (
                len(target_facility.ports[1]) == 3
                and len(target_facility.ports[-1]) == 1
            )
            if source_is_splitter and target_is_merger:
                model.Add(sum(transit) >= 1)

        minimum_transit_cells = problem.edge_min_transit_cells[connection_id]
        if minimum_transit_cells:
            model.Add(sum(transit) >= minimum_transit_cells)

        connection_variables.append(ConnectionVariables(arcs, transit, output_choice, input_choice))

    for choices in port_choices.values():
        if len(choices) > 1:
            model.AddAllDifferent(choices)

    # A physical grid edge can carry only one route, regardless of transport
    # kind. This also prevents two straight routes at a crossing from sharing
    # the same orientation.
    for u, v in undirected_edges:
        model.Add(
            sum(
                connection.arcs[arc_index[(u, v)]] + connection.arcs[arc_index[(v, u)]]
                for connection in connection_variables
            )
            <= 1
        )

    # A shared ordinary cell is a crossing only when exactly two routes pass
    # straight through it. Edge exclusivity above then forces those two routes
    # to use different (horizontal/vertical) orientations.
    for node in range(cell_count):
        transit_count = sum(connection.transit[node] for connection in connection_variables)
        model.Add(transit_count <= 2)
        is_crossing = model.NewBoolVar(f"cell_{node}_is_crossing")
        model.Add(transit_count == 2).OnlyEnforceIf(is_crossing)
        model.Add(transit_count <= 1).OnlyEnforceIf(is_crossing.Not())

        node_row, node_column = divmod(node, problem.columns)
        for connection_var in connection_variables:
            for incoming_arc in incoming[node]:
                incoming_source, _ = directed_arcs[incoming_arc]
                source_row, source_column = divmod(incoming_source, problem.columns)
                for outgoing_arc in outgoing[node]:
                    _, outgoing_target = directed_arcs[outgoing_arc]
                    target_row, target_column = divmod(outgoing_target, problem.columns)
                    is_straight = (
                        source_row + target_row == 2 * node_row
                        and source_column + target_column == 2 * node_column
                    )
                    if not is_straight:
                        model.Add(
                            connection_var.arcs[incoming_arc] + connection_var.arcs[outgoing_arc] <= 1
                        ).OnlyEnforceIf(is_crossing)

    model.Minimize(sum(variable for connection in connection_variables for variable in connection.arcs))
    return ModelArtifacts(model, facility_variables, connection_variables, directed_arcs, undirected_edges)


def build_solution(problem: Problem, artifacts: ModelArtifacts, solver: Any) -> dict[str, object]:
    machs: list[dict[str, object]] = []
    for facility_id, (facility, variables) in enumerate(zip(problem.facilities, artifacts.facilities)):
        row = solver.Value(variables.row)
        column = solver.Value(variables.column)
        rotation = solver.Value(variables.rotation)
        ports = [
            {
                "kind": kind,
                "orig": list(port),
                "cell": [row, column],
                "dir": (port[0] + rotation) % 4,
            }
            for kind in PORT_KINDS
            for port in facility.ports[kind]
        ]
        machs.append(
            {
                "id": facility_id,
                "type": "logi",
                "size": [1, 1],
                "pos": [row, column],
                "rot": rotation,
                "p": False,
                "occ_h": 1,
                "occ_w": 1,
                "ports": ports,
            }
        )

    def selected_port(facility_id: int, kind: int, choice_variable: Any) -> dict[str, object]:
        candidate_index = solver.Value(choice_variable)
        original = problem.facilities[facility_id].ports[kind][candidate_index]
        variables = artifacts.facilities[facility_id]
        rotation = solver.Value(variables.rotation)
        return {
            "index": candidate_index,
            "kind": kind,
            "orig": list(original),
            "cell": [solver.Value(variables.row), solver.Value(variables.column)],
            "dir": (original[0] + rotation) % 4,
        }

    belts: list[dict[str, object]] = []
    arc_index = {arc: index for index, arc in enumerate(artifacts.directed_arcs)}
    for connection_id, (connection, connection_vars) in enumerate(
        zip(problem.connections, artifacts.connections)
    ):
        edges = []
        for u, v in artifacts.undirected_edges:
            if solver.Value(connection_vars.arcs[arc_index[(u, v)]]) or solver.Value(connection_vars.arcs[arc_index[(v, u)]]):
                edges.append([list(divmod(u, problem.columns)), list(divmod(v, problem.columns))])
        belts.append(
            {
                "type": connection.kind,
                "frm": connection.source,
                "to": connection.target,
                "edges": edges,
                "min_transit_cells": problem.edge_min_transit_cells[connection_id],
                "transit_cells": max(0, len(edges) - 1),
                "port_assignment": {
                    "frm": selected_port(connection.source, -connection.kind, connection_vars.output_choice),
                    "to": selected_port(connection.target, connection.kind, connection_vars.input_choice),
                },
            }
        )

    input_snapshot = {
        key: problem.raw[key]
        for key in ("mach", "belt", "n", "m", "_editor")
        if key in problem.raw
    }
    return {
        "n": problem.rows,
        "m": problem.columns,
        "machs": machs,
        "belts": belts,
        "problem": problem.metadata,
        "input": input_snapshot,
    }


def resolve_output_path(output_dir: Path, requested_name: str | None, problem: Problem) -> Path:
    stem = requested_name or problem.metadata.get("name") or problem.fallback_name or "logistics"
    safe_stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "logistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{safe_stem}_solution.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{safe_stem}_solution_{suffix}.json"
        suffix += 1
    return candidate


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Endfield splitter/merger placement solver")
    parser.add_argument("--input", "-i", type=Path, help="editor-exported .aic.json; omit for the built-in demo")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."), help="solution output directory")
    parser.add_argument("--name", help="override the solution filename prefix")
    parser.add_argument("--time-limit", type=float, default=300.0, help="maximum solve time in seconds")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT search workers")
    parser.add_argument(
        "--search-mode",
        choices=("fast", "balanced", "optimal"),
        default="balanced",
        help="CP-SAT search policy (default: balanced)",
    )
    parser.add_argument("--debug", action="store_true", help="show CP-SAT search progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.time_limit <= 0 or args.workers <= 0:
        print("error: --time-limit and --workers must be positive", file=sys.stderr)
        return 2

    try:
        if args.input is None:
            raw = json.loads(json.dumps(DEFAULT_PROBLEM))
            fallback_name = "logistics_demo"
        else:
            raw = json.loads(args.input.read_text(encoding="utf-8"))
            fallback_name = args.input.name.removesuffix(".aic.json") or args.input.stem
        if not isinstance(raw, dict):
            raise ValueError("input JSON must contain an object at the top level")
        problem = parse_problem(raw, fallback_name)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    from ortools.sat.python import cp_model

    artifacts = build_model(problem, cp_model)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_search_workers = args.workers
    solver.parameters.log_search_progress = args.debug
    if args.search_mode == "fast":
        solver.parameters.stop_after_first_solution = True
    elif args.search_mode == "balanced":
        solver.parameters.relative_gap_limit = 0.05
    status = solver.Solve(artifacts.model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        label = {
            cp_model.INFEASIBLE: "NO (infeasible)",
            cp_model.MODEL_INVALID: "Invalid model",
            cp_model.UNKNOWN: "TL (no solution before timeout)",
        }.get(status, f"Solver status {status}")
        print(label)
        return 1

    solution = build_solution(problem, artifacts, solver)
    output_path = resolve_output_path(args.output_dir, args.name, problem)
    output_path.write_text(json.dumps(solution, ensure_ascii=False, indent=2), encoding="utf-8")

    label = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    print(f"OK ({label}), route length = {int(solver.ObjectiveValue())}")
    machs = solution["machs"]
    assert isinstance(machs, list)
    for machine in machs:
        assert isinstance(machine, dict)
        print(f"  #{machine['id']}: pos={machine['pos']}, rot={machine['rot'] * 90} deg")
    print(f"Solution exported to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
