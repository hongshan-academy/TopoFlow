"""Benchmark the native Rust flow solvers against the exact constructor certificate.

The graph corpus is built with the exact constructor, so every graph has a known
exact per-edge flow (a full-rank certificate).  For each graph we time the Rust
backends and compare both the boundary flow and the per-edge flows.

Run:  uv run python -m benchmarks.benchmark_flow_solvers
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from constructor import boundary_flow, construct_fraction

import topoflow_native


@dataclass(frozen=True, slots=True)
class Timing:
    backend: str
    samples: int
    total_seconds: float
    median_ms: float
    p95_ms: float


def fraction_corpus(max_denominator: int, extra: list[tuple[int, int]]) -> list[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for q in range(2, max_denominator + 1):
        for p in range(1, q):
            if math.gcd(p, q) == 1:
                targets.add((p, q))
    targets.update(extra)
    return sorted(targets)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def native_edge_fractions(result: object) -> list[Fraction] | None:
    """Exact fractions from the native solution, if the build exposes them."""
    numerators = getattr(result, "flow_numerators", None)
    denominators = getattr(result, "flow_denominators", None)
    if numerators is not None and denominators is not None:
        return [Fraction(int(n), int(d)) for n, d in zip(numerators, denominators)]
    return None


def native_edge_floats(result: object) -> list[float]:
    return [float(value) for value in result.flows]  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-denominator", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--extra",
        default="28/39,325/799,3020/3333",
        help="comma separated p/q targets added to the corpus",
    )
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("output/rust_flow_benchmark.json"))
    args = parser.parse_args()

    extra: list[tuple[int, int]] = []
    for item in args.extra.split(","):
        item = item.strip()
        if not item:
            continue
        p, q = item.split("/")
        extra.append((int(p), int(q)))

    targets = fraction_corpus(args.max_denominator, extra)
    names = ("rust-rank-smt", "rust-stable-polynomial")
    timings: dict[str, list[float]] = {name: [] for name in names}
    backend_counts: Counter[str] = Counter()
    mismatches: list[dict[str, object]] = []
    edge_ok: Counter[str] = Counter()
    edge_total: Counter[str] = Counter()

    graphs = 0
    for p, q in targets:
        result = construct_fraction(p, q)
        certificate = result.certificate
        edges = list(certificate.edges)
        expected_total = boundary_flow(certificate)
        expected_edges = list(certificate.flows)
        graphs += 1

        calls = {
            "rust-rank-smt": lambda: topoflow_native.solve_rank_smt(edges),
            "rust-stable-polynomial": lambda: topoflow_native.solve_stable_polynomial(
                edges, args.max_iterations, args.tolerance
            ),
        }
        for name, call in calls.items():
            try:
                native = call()
            except Exception as error:  # noqa: BLE001 - benchmark should keep going
                mismatches.append({"target": f"{p}/{q}", "backend": name, "error": str(error)})
                continue
            backend_counts[native.backend] += 1
            total_fraction = Fraction(
                int(native.total_numerator), int(native.total_denominator)
            )
            if total_fraction != expected_total:
                mismatches.append({
                    "target": f"{p}/{q}",
                    "backend": name,
                    "expected": str(expected_total),
                    "total": native.total_flow,
                    "total_fraction": str(total_fraction),
                })
            exact_edges = native_edge_fractions(native)
            if exact_edges is not None:
                for got, want in zip(exact_edges, expected_edges):
                    edge_total[name] += 1
                    if got == want:
                        edge_ok[name] += 1
            for _ in range(args.repeats):
                started = time.perf_counter()
                call()
                timings[name].append(time.perf_counter() - started)

    rows = []
    for name in names:
        values = timings[name]
        if not values:
            continue
        rows.append(Timing(
            name,
            len(values),
            sum(values),
            statistics.median(values) * 1000,
            percentile(values, 0.95) * 1000,
        ))

    report = {
        "graphs": graphs,
        "repeats": args.repeats,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "backend_counts": dict(sorted(backend_counts.items())),
        "edge_accuracy": {
            name: {"correct": edge_ok[name], "total": edge_total[name]}
            for name in names
        },
        "timings": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
