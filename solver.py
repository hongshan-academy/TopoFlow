from typing import List, Optional

from result import EdgeResult, SolverResult, format_result, visualize_result
from graph import Graph

import pulp


def solve(graph: Graph, msg: bool = False, threads: Optional[int] = None, exclude_patterns: Optional[List[List[bool]]] = None) -> SolverResult:
    (
        nodes, 
        edges, 
        out_edges, 
        in_edges, 
        sources, 
        sinks, 
        splitters, 
        convergers
    ) = (
        graph.nodes, 
        graph.edges, 
        graph.out_edges, 
        graph.in_edges, 
        graph.sources, 
        graph.sinks, 
        graph.splitters, 
        graph.convergers
    )
    if not sources or not sinks:
        return SolverResult(pulp.constants.LpStatusUndefined)
    
    # main problem
    problem = pulp.LpProblem("topoflow_system", pulp.LpMaximize)
    
    # A x = b
    # x -> L  (if blocked)
    #      H  (if not blocked)

    v = pulp.LpVariable.dicts('v', edges, lowBound=0, upBound=1, cat=pulp.const.LpContinuous)
    
    v_H_in  = pulp.LpVariable.dicts('v_H_in', edges, lowBound=0, upBound=1, cat=pulp.const.LpContinuous)
    v_L_in  = pulp.LpVariable.dicts('v_L_in', edges, lowBound=0, upBound=1, cat=pulp.const.LpContinuous)
    v_H_out = pulp.LpVariable.dicts('v_H_out', edges, lowBound=0, upBound=1, cat=pulp.const.LpContinuous)
    v_L_out = pulp.LpVariable.dicts('v_L_out', edges, lowBound=0, upBound=1, cat=pulp.const.LpContinuous)

    is_blocked = pulp.LpVariable.dicts('is_blocked', edges, cat=pulp.const.LpBinary)
    is_full = pulp.LpVariable.dicts('is_full', edges, cat=pulp.const.LpBinary)
    
    # conservation
    for node in nodes:
        if node in (*sources, *sinks):
            continue
        
        incoming_edges = in_edges[node]
        outgoing_edges = out_edges[node]
        problem += (
            pulp.lpSum(v[edge] for edge in incoming_edges) == 
            pulp.lpSum(v[edge] for edge in outgoing_edges)
        )
    
    # branch linearization
    # x -> L  (if blocked)
    #      H  (if not blocked)
    M = 2
    for edge in edges:
        s = is_blocked[edge]
        
        # full
        problem += v[edge] >= is_full[edge]
        problem += v_L_in[edge] >= is_full[edge]
        problem += v_H_in[edge] >= is_full[edge]
        problem += v_L_out[edge] >= is_full[edge]
        problem += v_H_out[edge] >= is_full[edge]

        # splitter
        # blocked -> L
        problem += v[edge] >= v_L_in[edge] - M * (1 - s)
        problem += v[edge] <= v_L_in[edge] + M * (1 - s)
        # not blocked -> H
        problem += v[edge] >= v_H_in[edge] - M * s
        problem += v[edge] <= v_H_in[edge] + M * s

        # converger
        # blocked -> H
        problem += v[edge] >= v_H_out[edge] - M * (1 - s)
        problem += v[edge] <= v_H_out[edge] + M * (1 - s)
        # not blocked -> L
        problem += v[edge] >= v_L_out[edge] - M * s
        problem += v[edge] <= v_L_out[edge] + M * s
    
    # # H > L
    # EPS = 1e-6
    
    # H >= L
    EPS = 0
    for edge in edges:
        problem += v_H_in[edge] - v_L_in[edge] >= EPS * (1 - is_full[edge])
        problem += v_H_out[edge] - v_L_out[edge] >= EPS * (1 - is_full[edge])
        
    # evenly split => shared variable
    for splitter_node in splitters:
        outgoing_edges = out_edges[splitter_node].copy()
        first_edge = outgoing_edges.pop()
        while outgoing_edges:
            edge = outgoing_edges.pop()
            problem += v_H_in[edge] == v_H_in[first_edge]
            # problem += v_L_in[edge] == v_L_in[first_edge]

    for converger_node in convergers:
        incoming_edges = in_edges[converger_node].copy()
        first_edge = incoming_edges.pop()
        while incoming_edges:
            edge = incoming_edges.pop()
            problem += v_H_out[edge] == v_H_out[first_edge]
            # problem += v_L_out[edge] == v_L_out[first_edge]
    
    # output should not be blocked
    for sink_node in sinks:
        for edge in in_edges[sink_node]:
            problem += is_blocked[edge] == 0
    for source_node in sources:
        for edge in out_edges[source_node]:
            problem += is_blocked[edge] == 1
    
    # blockage propagation
    for splitter_node in splitters:
        for incoming_edge in in_edges[splitter_node]:
            outgoing_edges = out_edges[splitter_node]
            
            # exists edge_out, edge_out is not blocked => edge_in is not blocked
            for outgoing_edge in outgoing_edges:
                problem += (
                    is_blocked[incoming_edge] <= is_blocked[outgoing_edge] + 
                    is_full[incoming_edge] # ignore full cases
                )
                
            # edge_in is not blocked => exists edge_out, edge_out is not blocked
            problem += (
                pulp.lpSum(is_blocked[edge] for edge in outgoing_edges) <= 
                len(outgoing_edges) - 1 + is_blocked[incoming_edge]
            )
            
    for converger_node in convergers:
        for outgoing_edge in out_edges[converger_node]:
            incoming_edges = in_edges[converger_node]
            
            # exists edge_in, edge_in is blocked => edge_out is blocked
            for incoming_edge in incoming_edges:
                problem += is_blocked[incoming_edge] <= (
                    is_blocked[outgoing_edge] + 
                    is_full[outgoing_edge] # ignore full cases
                )

            # edge_out is blocked => exists edge_in, edge_in is blocked
            problem += (
                is_blocked[outgoing_edge] <= 
                pulp.lpSum(is_blocked[edge] for edge in incoming_edges)
            )

    # exclude already-known solutions
    if exclude_patterns:
        for pattern in exclude_patterns:
            terms = []
            for edge_index, edge in enumerate(edges):
                if pattern[edge_index]:
                    terms.append(1 - is_blocked[edge])
                else:
                    terms.append(is_blocked[edge])
            problem += pulp.lpSum(terms) >= 1

    # run
    # f_min = pulp.LpVariable('f_min', lowBound=0, upBound=1, cat=pulp.const.LpContinuous)
    # for edge in edges:
    #     problem += v[edge] >= f_min
    # problem += f_min
    # problem += pulp.lpSum(is_full)
    problem += 0

    problem.solve(pulp.HiGHS(msg=msg, threads=threads))


    result = SolverResult(
        problem.status,
        [EdgeResult(
            edge[0], edge[1], 
            pulp.value(v[edge]), 
            bool(pulp.value(is_blocked[edge])),
            bool(pulp.value(is_full[edge]))) 
         for edge in edges]
    )

    return result

if __name__ == '__main__':
    # for data in ['graph/graph_0.4.txt', 'graph/graph_0.375.txt', 'graph/graph_0.333.txt', 'graph/graph_0.5.txt', 'graph/graph_0.333_simple.txt']:
    # for data in ['graph/graph.txt']:
    # for data in ['graph/graph_2sinks.txt']:
    for data in ['output/ga_output_1.txt']:
        with open(data, 'r', encoding='utf-8') as file:
            result = solve(Graph.from_text(file.read()), msg=True)
            print(format_result(result))
        # with open('output/visualize/graph_0.333.dot', 'w', encoding='utf-8') as file:
        #     file.write(visualize_result(result))

