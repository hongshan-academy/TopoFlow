"""Physical layout solver (OR-Tools CP-SAT).

Constraint-programming facility placement and belt routing for TopoFlow graphs.
The algorithm and original implementation are by @kokobird (TopoFlow-CLI).
"""

from .adapter import (
    AdaptedTopoFlowProblem,
    NodeTypeMismatchError,
    SolveFailure,
    adapt_layout_request,
    adapt_topoflow,
    solve_topoflow,
    solve_with_expansion,
)
from .solver import (
    ModelArtifacts,
    Problem,
    build_model,
    build_solution,
    parse_problem,
    resolve_output_path,
)

__all__ = [
    "AdaptedTopoFlowProblem",
    "ModelArtifacts",
    "NodeTypeMismatchError",
    "Problem",
    "SolveFailure",
    "adapt_layout_request",
    "adapt_topoflow",
    "build_model",
    "build_solution",
    "parse_problem",
    "resolve_output_path",
    "solve_topoflow",
    "solve_with_expansion",
]
