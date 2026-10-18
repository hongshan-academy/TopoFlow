from typing import List, Optional

from result import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNDEFINED,
    EdgeResult,
    SolverResult,
)
from graph import Graph, control_gear

from ilpbridge import FEASIBLE, INFEASIBLE, OPTIMAL, UNKNOWN, Model, Solver, _LinearExpr

_ENGINE_KEYS = {"rank-smt", "karzanov"}


_MINGW_BIN = r"C:\msys64\ucrt64\bin"


def solve_native(
    engine: str, nodes: List[dict], edges: List[dict], workers: int = 16
) -> dict:
    """Exact rational solve through the Rust extension (karzanov / rank-smt)."""
    import os

    if os.name == "nt" and os.path.isdir(_MINGW_BIN):
        try:
            os.add_dll_directory(_MINGW_BIN)
        except OSError:
            pass
    try:
        import topoflow_native
    except ImportError as e:
        raise RuntimeError(f"Rust solver unavailable ({e}); run `uv sync` first to build the extension") from e

    if engine not in _ENGINE_KEYS:
        raise ValueError(f"Unknown solver engine: {engine}")
    node_types: dict[str, str] = {}
    for node in nodes:
        node_id = node["id"]
        if node_id in node_types:
            raise ValueError(f"Duplicate node ID: {node_id}")
        node_types[node_id] = node["type"]

    source_ids = {node_id for node_id, node_type in node_types.items() if node_type == "In"}
    sink_ids = {node_id for node_id, node_type in node_types.items() if node_type == "Out"}
    if not source_ids:
        raise ValueError("Solver requires at least one input node")
    if not sink_ids:
        raise ValueError("Solver requires at least one output node")

    degrees = {node_id: [0, 0] for node_id in node_types}
    edge_pairs = []
    for edge in edges:
        source, target = edge["from"], edge["to"]
        if source not in degrees or target not in degrees:
            raise ValueError(f"Edge {edge.get('id', '<unknown>')} references an unknown node")
        degrees[source][1] += 1
        degrees[target][0] += 1
        edge_pairs.append((source, target))

    for node_id in source_ids:
        if degrees[node_id] != [0, 1]:
            raise ValueError(f"Input node {node_id} must have degree (0, 1), got {tuple(degrees[node_id])}")
    for node_id in sink_ids:
        if degrees[node_id] != [1, 0]:
            raise ValueError(f"Output node {node_id} must have degree (1, 0), got {tuple(degrees[node_id])}")

    inferred_sources = {node_id for node_id, degree in degrees.items() if degree == [0, 1]}
    inferred_sinks = {node_id for node_id, degree in degrees.items() if degree == [1, 0]}
    if inferred_sources != source_ids:
        raise ValueError("Every node with degree (0, 1) must be declared In, and no In node may have another degree")
    if inferred_sinks != sink_ids:
        raise ValueError("Every node with degree (1, 0) must be declared Out, and no Out node may have another degree")
    if engine == "rank-smt":
        res = topoflow_native.solve_rank_smt(edge_pairs, None, workers)
    else:
        res = topoflow_native.solve_karzanov(edge_pairs, 10000, 1e-10)
    numerators = [int(value) for value in res.flow_numerators]
    denominators = [int(value) for value in res.flow_denominators]
    total_numerator = int(res.total_numerator)
    total_denominator = int(res.total_denominator)
    can_in = list(getattr(res, "can_in", None) or [])
    can_out = list(getattr(res, "can_out", None) or [])
    has_states = len(can_in) == len(edges) == len(can_out)

    edge_flows: List[dict] = []
    for index, (edge, value, numerator, denominator) in enumerate(
        zip(edges, res.flows, numerators, denominators)
    ):
        entry = {
            "from": edge["from"],
            "to": edge["to"],
            "flow": float(value),
            "numerator": numerator,
            "denominator": denominator,
            "text": f"{numerator}/{denominator}",
        }
        if has_states:
            # normal: receives only; blocked: delivers only; full: both or saturated.
            full = (can_in[index] and can_out[index]) or numerator == denominator
            state = "full" if full else "blocked" if can_out[index] else "normal"
            entry["state"] = state
            entry["isBlocked"] = state == "blocked"
            entry["isFull"] = full
        edge_flows.append(entry)

    return {
        "model": engine,
        "backend": res.backend,
        "status": res.status,
        "feasible": res.feasible,
        "totalFlow": res.total_flow,
        "totalNumerator": total_numerator,
        "totalDenominator": total_denominator,
        "totalText": f"{total_numerator}/{total_denominator}",
        "iterations": res.iterations,
        "edgeFlows": edge_flows,
    }


class FlowModel:
    """Exact MILP flow model, reusable across exclusion-constrained solves.

    Building the variable/constraint set is the expensive part; callers that
    enumerate several solutions can add exclusion patterns and re-solve the
    same model instead of rebuilding it from scratch each time.
    """

    def __init__(self, graph: Graph, workers: int = 16) -> None:
        (
            nodes,
            edges,
            out_edges,
            in_edges,
            sources,
            sinks,
            splitters,
            convergers,
        ) = (
            graph.nodes,
            graph.edges,
            graph.out_edges,
            graph.in_edges,
            graph.sources,
            graph.sinks,
            # Control units are degree-(1, 1) nodes that the graph classifies as
            # splitters; they get their own throttling rules below.
            [node for node in graph.splitters if control_gear(node) is None],
            [node for node in graph.convergers if control_gear(node) is None],
        )
        self.edges = edges
        self._undefined = not sources or not sinks
        if self._undefined:
            return

        # Exact MILP built on the shared Z3 bridge:
        #   A x = b,  x -> L (if blocked) / H (if unblocked); both = full
        model = Model()
        v = {edge: model.new_real_var(0, 1, f"v_{index}") for index, edge in enumerate(edges)}
        v_H_in = {edge: model.new_real_var(0, 1, f"v_H_in_{index}") for index, edge in enumerate(edges)}
        v_L_in = {edge: model.new_real_var(0, 1, f"v_L_in_{index}") for index, edge in enumerate(edges)}
        v_H_out = {edge: model.new_real_var(0, 1, f"v_H_out_{index}") for index, edge in enumerate(edges)}
        v_L_out = {edge: model.new_real_var(0, 1, f"v_L_out_{index}") for index, edge in enumerate(edges)}

        is_blocked = {edge: model.new_bool_var(f"is_blocked_{index}") for index, edge in enumerate(edges)}
        is_unblocked = {edge: model.new_bool_var(f"is_unblocked_{index}") for index, edge in enumerate(edges)}

        # Conservation at every internal node.
        for node in nodes:
            if node in (*sources, *sinks):
                continue
            model.add(
                sum(v[edge] for edge in in_edges[node])
                == sum(v[edge] for edge in out_edges[node])
            )

        # Branch linearization: x -> L (if blocked) / H (if unblocked).
        M = 2
        for edge in edges:
            b = is_blocked[edge]
            ub = is_unblocked[edge]

            # No-idle: every edge is blocked, unblocked, or both.
            model.add(b + ub >= 1)
            # Full: both blocked and unblocked => v = 1.
            model.add(v[edge] >= b + ub - 1)

            # Splitter: blocked -> L.
            model.add(v[edge] >= v_L_in[edge] - M * (1 - b))
            model.add(v[edge] <= v_L_in[edge] + M * (1 - b))
            # Splitter: unblocked -> H.
            model.add(v[edge] >= v_H_in[edge] - M * (1 - ub))
            model.add(v[edge] <= v_H_in[edge] + M * (1 - ub))

            # Converger: blocked -> H.
            model.add(v[edge] >= v_H_out[edge] - M * (1 - b))
            model.add(v[edge] <= v_H_out[edge] + M * (1 - b))
            # Converger: unblocked -> L.
            model.add(v[edge] >= v_L_out[edge] - M * (1 - ub))
            model.add(v[edge] <= v_L_out[edge] + M * (1 - ub))

        # H >= L (equal split).
        for edge in edges:
            model.add(v_H_in[edge] - v_L_in[edge] >= 0)
            model.add(v_H_out[edge] - v_L_out[edge] >= 0)

        # Evenly split => shared variable.
        for splitter_node in splitters:
            outgoing_edges = out_edges[splitter_node].copy()
            first_edge = outgoing_edges.pop()
            while outgoing_edges:
                edge = outgoing_edges.pop()
                model.add(v_H_in[edge] == v_H_in[first_edge])

        for converger_node in convergers:
            incoming_edges = in_edges[converger_node].copy()
            first_edge = incoming_edges.pop()
            while incoming_edges:
                edge = incoming_edges.pop()
                model.add(v_H_out[edge] == v_H_out[first_edge])

        # Output should be unblocked; input must be blocked.
        for sink_node in sinks:
            for edge in in_edges[sink_node]:
                model.add(is_unblocked[edge] == 1)
        for source_node in sources:
            for edge in out_edges[source_node]:
                model.add(is_blocked[edge] == 1)

        # Blockage propagation.
        for splitter_node in splitters:
            for incoming_edge in in_edges[splitter_node]:
                outgoing_edges = out_edges[splitter_node]

                # Exists edge_out, edge_out is unblocked => edge_in is unblocked.
                for outgoing_edge in outgoing_edges:
                    model.add(is_unblocked[incoming_edge] >= is_unblocked[outgoing_edge])

                # Edge_in is unblocked => exists edge_out, edge_out is unblocked.
                model.add(
                    is_unblocked[incoming_edge]
                    <= sum(is_unblocked[edge] for edge in outgoing_edges)
                    + is_blocked[incoming_edge]
                )

        for converger_node in convergers:
            for outgoing_edge in out_edges[converger_node]:
                incoming_edges = in_edges[converger_node]

                # Exists edge_in, edge_in is blocked => edge_out is blocked.
                for incoming_edge in incoming_edges:
                    model.add(is_blocked[incoming_edge] <= is_blocked[outgoing_edge])

                # Edge_out is blocked => exists edge_in, edge_in is blocked.
                model.add(
                    is_blocked[outgoing_edge]
                    <= sum(is_blocked[edge] for edge in incoming_edges)
                    + is_unblocked[outgoing_edge]
                )

        # Control units throttle their single incoming edge to ``[0, gear/5]``
        # and only ever propagate blockage from downstream to upstream.
        for node in nodes:
            gear = control_gear(node)
            if gear is None:
                continue
            in_edge = in_edges[node][0]
            out_edge = out_edges[node][0]

            # Capacity: the incoming edge's full speed is gear/5.
            model.add(v[in_edge] * 5 <= gear)

            # Downstream blocked => upstream blocked (equivalently, upstream
            # flowing => downstream flowing).
            model.add(is_blocked[out_edge] <= is_blocked[in_edge])
            model.add(is_unblocked[in_edge] <= is_unblocked[out_edge])

            # Upstream blocked => incoming flow saturates at gear/5.
            model.add(v[in_edge] * 5 >= gear - 10 + 10 * is_blocked[in_edge])

        self.model = model
        self.v = v
        self.is_blocked = is_blocked
        self.is_unblocked = is_unblocked
        self._solver = Solver()
        self._solver.parameters.max_time_in_seconds = 30.0
        self._solver.parameters.num_search_workers = workers

    def add_exclusion_pattern(self, pattern: List[bool]) -> None:
        """Forbid the exact blocked/unblocked assignment described by ``pattern``."""
        expression = _LinearExpr(self.model)
        for edge_index, edge in enumerate(self.edges):
            if pattern[edge_index]:
                expression.constant += 1
                expression.add_var(self.is_blocked[edge], -1)
            else:
                expression.add_var(self.is_blocked[edge], 1)
        self.model.add(expression >= 1)

    def solve(self) -> SolverResult:
        if self._undefined:
            return SolverResult(STATUS_UNDEFINED)
        status = self._solver.solve(self.model)
        if status == INFEASIBLE:
            return SolverResult(STATUS_INFEASIBLE)
        if status not in (OPTIMAL, FEASIBLE):
            return SolverResult(STATUS_UNDEFINED)
        return SolverResult(
            STATUS_OPTIMAL,
            [
                EdgeResult(
                    edge[0],
                    edge[1],
                    float(self._solver.value(self.v[edge])),
                    self._solver.int_value(self.is_blocked[edge]) > 0,
                    self._solver.int_value(self.is_unblocked[edge]) > 0,
                )
                for edge in self.edges
            ],
        )


def solve(
    graph: Graph,
    exclude_patterns: Optional[List[List[bool]]] = None,
    workers: int = 16,
) -> SolverResult:
    flow = FlowModel(graph, workers)
    for pattern in exclude_patterns or []:
        flow.add_exclusion_pattern(pattern)
    return flow.solve()
