from typing import List, Optional

from result import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNDEFINED,
    EdgeResult,
    SolverResult,
    format_result,
    visualize_result,
)
from graph import Graph

from ilpbridge import FEASIBLE, INFEASIBLE, OPTIMAL, UNKNOWN, Model, Solver, _LinearExpr


def solve(
    graph: Graph,
    msg: bool = False,
    threads: Optional[int] = None,
    exclude_patterns: Optional[List[List[bool]]] = None,
) -> SolverResult:
    del msg, threads  # accepted for API compatibility; the exact bridge is single-threaded
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
        graph.splitters,
        graph.convergers,
    )
    if not sources or not sinks:
        return SolverResult(STATUS_UNDEFINED)

    # Exact MILP built on the shared Z3 bridge:
    #   A x = b,  x -> L (if blocked) / H (if not blocked)
    model = Model()
    v = {edge: model.new_real_var(0, 1, f"v_{index}") for index, edge in enumerate(edges)}
    v_H_in = {edge: model.new_real_var(0, 1, f"v_H_in_{index}") for index, edge in enumerate(edges)}
    v_L_in = {edge: model.new_real_var(0, 1, f"v_L_in_{index}") for index, edge in enumerate(edges)}
    v_H_out = {edge: model.new_real_var(0, 1, f"v_H_out_{index}") for index, edge in enumerate(edges)}
    v_L_out = {edge: model.new_real_var(0, 1, f"v_L_out_{index}") for index, edge in enumerate(edges)}

    is_blocked = {edge: model.new_bool_var(f"is_blocked_{index}") for index, edge in enumerate(edges)}
    is_full = {edge: model.new_bool_var(f"is_full_{index}") for index, edge in enumerate(edges)}

    # Conservation at every internal node.
    for node in nodes:
        if node in (*sources, *sinks):
            continue
        model.add(
            sum(v[edge] for edge in in_edges[node])
            == sum(v[edge] for edge in out_edges[node])
        )

    # Branch linearization: x -> L (if blocked) / H (if not blocked).
    M = 2
    for edge in edges:
        s = is_blocked[edge]

        # Full.
        model.add(v[edge] >= is_full[edge])
        model.add(v_L_in[edge] >= is_full[edge])
        model.add(v_H_in[edge] >= is_full[edge])
        model.add(v_L_out[edge] >= is_full[edge])
        model.add(v_H_out[edge] >= is_full[edge])

        # Splitter: blocked -> L.
        model.add(v[edge] >= v_L_in[edge] - M * (1 - s))
        model.add(v[edge] <= v_L_in[edge] + M * (1 - s))
        # Not blocked -> H.
        model.add(v[edge] >= v_H_in[edge] - M * s)
        model.add(v[edge] <= v_H_in[edge] + M * s)

        # Converger: blocked -> H.
        model.add(v[edge] >= v_H_out[edge] - M * (1 - s))
        model.add(v[edge] <= v_H_out[edge] + M * (1 - s))
        # Not blocked -> L.
        model.add(v[edge] >= v_L_out[edge] - M * s)
        model.add(v[edge] <= v_L_out[edge] + M * s)

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

    # Output should not be blocked; input must be blocked.
    for sink_node in sinks:
        for edge in in_edges[sink_node]:
            model.add(is_blocked[edge] == 0)
    for source_node in sources:
        for edge in out_edges[source_node]:
            model.add(is_blocked[edge] == 1)

    # Blockage propagation.
    for splitter_node in splitters:
        for incoming_edge in in_edges[splitter_node]:
            outgoing_edges = out_edges[splitter_node]

            # Exists edge_out, edge_out is not blocked => edge_in is not blocked.
            for outgoing_edge in outgoing_edges:
                model.add(is_blocked[incoming_edge] <= is_blocked[outgoing_edge] + is_full[incoming_edge])

            # Edge_in is not blocked => exists edge_out, edge_out is not blocked.
            model.add(
                sum(is_blocked[edge] for edge in outgoing_edges)
                <= len(outgoing_edges) - 1 + is_blocked[incoming_edge]
            )

    for converger_node in convergers:
        for outgoing_edge in out_edges[converger_node]:
            incoming_edges = in_edges[converger_node]

            # Exists edge_in, edge_in is blocked => edge_out is blocked.
            for incoming_edge in incoming_edges:
                model.add(is_blocked[incoming_edge] <= is_blocked[outgoing_edge] + is_full[outgoing_edge])

            # Edge_out is blocked => exists edge_in, edge_in is blocked.
            model.add(is_blocked[outgoing_edge] <= sum(is_blocked[edge] for edge in incoming_edges))

    # Exclude already-known solutions.
    if exclude_patterns:
        for pattern in exclude_patterns:
            expression = _LinearExpr(model)
            for edge_index, edge in enumerate(edges):
                if pattern[edge_index]:
                    expression.constant += 1
                    expression.add_var(is_blocked[edge], -1)
                else:
                    expression.add_var(is_blocked[edge], 1)
            model.add(expression >= 1)

    solver = Solver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(model)
    if status == INFEASIBLE:
        return SolverResult(STATUS_INFEASIBLE)
    if status not in (OPTIMAL, FEASIBLE):
        return SolverResult(STATUS_UNDEFINED)

    result = SolverResult(
        STATUS_OPTIMAL,
        [
            EdgeResult(
                edge[0],
                edge[1],
                float(solver.value(v[edge])),
                solver.int_value(is_blocked[edge]) > 0,
                solver.int_value(is_full[edge]) > 0,
            )
            for edge in edges
        ],
    )
    return result


if __name__ == '__main__':
    for data in ['output/ga_output_1.txt']:
        with open(data, 'r', encoding='utf-8') as file:
            result = solve(Graph.from_text(file.read()), msg=True)
            print(format_result(result))
