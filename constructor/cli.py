"""Command-line interface for universal rational-flow construction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .leaves.unit import DEFAULT_EXHAUSTION_PAIR_LIMIT
from .service.construction import construct_fraction
from .verification.karzanov import crosscheck_karzanov


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct an exact TopoFlow graph with boundary flow p/q."
    )
    parser.add_argument("p", type=int)
    parser.add_argument("q", type=int)
    parser.add_argument(
        "--unit-solver",
        choices=("mixed", "exhaustion", "milp"),
        default="mixed",
        help="backend for odd 1/m modules (default: mixed)",
    )
    parser.add_argument(
        "--exhaustion-pair-limit",
        type=int,
        default=DEFAULT_EXHAUSTION_PAIR_LIMIT,
        help="maximum exhaustive prefix per mixed-mode layer (default: 300000)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        help="stop odd unit search after this feedback-permutation length",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="search for a smaller interval topology before using the guaranteed interval construction",
    )
    parser.add_argument("--search-range", type=int, default=0)
    parser.add_argument(
        "--reduction-depth",
        type=int,
        default=3,
        help="maximum smart-reduction nesting depth (default: 3)",
    )
    parser.add_argument(
        "--reduction-state-limit",
        type=int,
        default=4096,
        help="maximum memoized smart-reduction states (default: 4096)",
    )
    parser.add_argument("--output", type=Path, help="write one graph edge per line")
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="finally verify the graph with the bundled Rust Karzanov solver",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="show construction state; repeat for debug-level search states",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    log_level = (
        logging.DEBUG
        if args.verbose > 1
        else logging.INFO
        if args.verbose
        else logging.WARNING
    )
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = construct_fraction(
            args.p,
            args.q,
            max_k=args.max_k,
            optimize=args.optimize,
            search_range=args.search_range,
            unit_solver=args.unit_solver,
            exhaustion_pair_limit=args.exhaustion_pair_limit,
            reduction_depth=args.reduction_depth,
            reduction_state_limit=args.reduction_state_limit,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    plan = result.reduction
    certificate = result.certificate
    check = result.validation
    print(f"target={result.target}")
    print(f"strategy={plan.strategy} cost={plan.cost.nodes}n/{plan.cost.edges}e")
    stats = plan.stats
    print(
        f"search=depth:{args.reduction_depth} states:{stats.expanded_states} "
        f"candidates:{stats.candidate_count} pruned:{stats.pruned_candidates} "
        f"unit-resolutions:{stats.unit_resolutions} iterations:{stats.iterations}"
    )
    print("steps=" + " | ".join(plan.steps()))
    print(
        f"nodes={certificate.node_count} edges={certificate.edge_count} "
        f"fixed={len(certificate.fixed)} rank={check.rank}/{check.edge_count}"
    )
    timing = result.timing
    print(
        f"timing=planning:{timing.planning_seconds:.6f}s "
        f"unit-search:{timing.unit_search_seconds:.6f}s "
        f"realization:{timing.realization_seconds:.6f}s "
        f"validation:{timing.validation_seconds:.6f}s "
        f"total:{timing.total_seconds:.6f}s"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(f"{source} -> {target}\n" for source, target in certificate.edges),
            encoding="utf-8",
        )
        print(f"wrote {args.output.resolve()}")
    if args.cross_check:
        checked = crosscheck_karzanov(certificate, result.target)
        print(
            f"cross-check={checked.backend} status={checked.status} "
            f"flow={checked.flow} iterations={checked.iterations} "
            f"elapsed:{checked.elapsed_seconds:.6f}s"
        )


if __name__ == "__main__":
    main()
