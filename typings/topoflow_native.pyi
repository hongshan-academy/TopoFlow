from typing import Sequence

class NativeFlowSolution:
    backend: str
    status: str
    feasible: bool
    flows: list[float]
    total_flow: float
    iterations: int
    flow_numerators: list[int]
    flow_denominators: list[int]
    total_numerator: int
    total_denominator: int
    full_rank: bool

def solve_rank_smt(
    edges: Sequence[tuple[str, str]],
    fixed_edges: Sequence[int] | None = ...,
    workers: int = ...,
) -> NativeFlowSolution: ...
def solve_karzanov(
    edges: Sequence[tuple[str, str]],
    max_iterations: int = ...,
    tolerance: float = ...,
) -> NativeFlowSolution: ...
def solve_ilp_exact(spec: str) -> str: ...
def native_version() -> str: ...
