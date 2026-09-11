from dataclasses import dataclass, field
from typing import Dict, List

# Solver status strings used by the exact MILP engine.
STATUS_OPTIMAL = "Optimal"
STATUS_INFEASIBLE = "Infeasible"
STATUS_UNDEFINED = "Undefined"


@dataclass
class EdgeResult:
    source: str
    target: str
    flow: float
    is_blocked: bool
    is_full: bool = False

@dataclass
class SolverResult:
    status: str
    edges: List[EdgeResult] = field(default_factory=list)

@dataclass
class SimulatorResult:
    edges: List[EdgeResult]
    converged: bool = True

def format_result(result: SolverResult) -> str:
    string = f'FESSIBLE: {result.status}\n\n'
    if result.status == STATUS_OPTIMAL:
        string += f'EDGES: \n'
        for edge_result in result.edges:
            if edge_result.is_blocked:
                string += f'  {edge_result.source} -> {edge_result.target}: v={edge_result.flow:.6f} [blocked]\n'
            else:
                string += f'  {edge_result.source} -> {edge_result.target}: v={edge_result.flow:.6f}\n'
        string += '\n'
    
    return string


def visualize_result(result: SolverResult) -> str:
    lines = ["digraph G {"]

    counter: Dict[str, List[int]] = {}
    for edge in result.edges:
        counter.setdefault(edge.source, [0, 0])
        counter.setdefault(edge.target, [0, 0])
        counter[edge.source][1] += 1
        counter[edge.target][0] += 1

    for node, (in_deg, out_deg) in counter.items():
        if in_deg == 0 or out_deg == 0:
            continue
        if in_deg < out_deg:
            lines.append(f'  "{node}" [fillcolor=orange, style=filled]')
        elif in_deg > out_deg:
            lines.append(f'  "{node}" [fillcolor=cadetblue, style=filled]')

    for edge in result.edges:
        color = "red" if edge.is_blocked else "black"
        lines.append(
            f'  "{edge.source}" -> "{edge.target}" '
            f'[label="v={edge.flow:.6f}", color={color}]'
        )

    lines.append("}")
    return "\n".join(lines)
