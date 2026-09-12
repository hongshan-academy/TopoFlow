from dataclasses import dataclass, field
from typing import List

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


@dataclass
class SolverResult:
    status: str
    edges: List[EdgeResult] = field(default_factory=list)
