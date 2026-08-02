"""
Collect optimal graphs for simple fraction splits.
Prioritizes prime denominators < 100, then numerator != 1 fractions.
Only saves verified err=0 solutions.
"""

import json
import math
import os
import time
from collections import OrderedDict

from config import DEFAULT_CONFIG
from ga.fitness import evaluate_cached
from ga.core import run as ga_run


# ── Patch DEFAULT_CONFIG for small/fast runs ──
DEFAULT_CONFIG.max_edges = 12
DEFAULT_CONFIG.internal_nodes_choices = list(range(5, 21))
DEFAULT_CONFIG.internal_nodes_weights = [
    math.exp(-((x - 10) ** 2) / (2 * 1.5 ** 2)) for x in range(5, 21)
]
DEFAULT_CONFIG.stagnation_interval = 8
DEFAULT_CONFIG.stagnation_restart = 30
DEFAULT_CONFIG.stagnation_boost_ratio = 0.40
DEFAULT_CONFIG.restart_survivors = 5
DEFAULT_CONFIG.seed_edges = []

# ── Params (from Run 7 calibration) ──
BASE_PARAMS = dict(
    pop_size=80,
    generations=100,
    mutation_rate=0.68,
    crossover_rate=0.55,
    tournament_size=3,
    elitism_count=5,
    immigration_rate=0.22,
    mutation_weights=(0.15, 0.15, 0.10, 0.60),
    eval_timeout=None,
    mode="mixed",
    surrogate_enabled=True,
    surrogate_top_fraction=0.12,
    surrogate_random_eval_fraction=0.10,
    solver_workers=16,
    solver_threads=1,
)

PRIME_DENOMS = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
]

# Non-trivial numerators to try for small denominators
EXTRA_TARGETS = [
    (2, 3), (2, 5), (3, 5), (4, 5),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (2, 11), (3, 11), (4, 11), (5, 11), (7, 11),
    (2, 13), (3, 13), (5, 13), (7, 13), (11, 13),
]


def search_one_target(num: int, den: int) -> dict | None:
    label = f"{num}/{den}"
    output_dir = f"output/collect/{num}_{den}"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ga_top5.json")
    hist_path = os.path.join(output_dir, "ga_history.json")

    # Set history path via DEFAULT_CONFIG (not a run() parameter)
    DEFAULT_CONFIG.history_path = hist_path

    print(f"\n{'='*50}")
    print(f"  Target: {label}")
    print(f"{'='*50}")

    t0 = time.perf_counter()
    try:
        results = ga_run(
            target_pq=(num, den),
            output_path=out_path,
            **BASE_PARAMS,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    if not results:
        print(f"  No results returned")
        return None

    # Verify each hof entry with actual evaluation
    best = None
    for entry in results:
        if entry.get("error") != 0:
            continue
        edges = entry["graph"]["edges"]
        edges_tup = tuple(sorted((e[0], e[1]) for e in edges))
        real_err, real_nodes = evaluate_cached(edges_tup, (num, den), mode="mixed")
        if abs(real_err) < 1e-15:
            if best is None or real_nodes < best["verified_nodes"]:
                best = {
                    "target": {"p": num, "q": den},
                    "nodes": entry["nodes"],
                    "edges_count": len(edges),
                    "edges": edges,
                    "verified_nodes": real_nodes,
                }

    if best:
        print(f"  VERIFIED: {best['nodes']}n/{best['edges_count']}e  err=0")
    else:
        print(f"  No verified err=0 solution found")

    return best


def main():
    os.makedirs("output/collect", exist_ok=True)
    collection: dict = OrderedDict()
    failed: list = []
    total_start = time.perf_counter()

    # ── Phase 1: Prime denominators (1/p) ──
    print("=" * 60)
    print("  PHASE 1: 1/p for primes p < 100")
    print("=" * 60)

    for p in PRIME_DENOMS:
        result = search_one_target(1, p)
        label = f"1/{p}"
        if result:
            collection[label] = result
        else:
            failed.append(label)

    # ── Phase 2: Numerator != 1 ──
    print("\n" + "=" * 60)
    print("  PHASE 2: numerator != 1 fractions")
    print("=" * 60)

    for num, den in EXTRA_TARGETS:
        label = f"{num}/{den}"
        if label in collection:
            continue
        result = search_one_target(num, den)
        if result:
            collection[label] = result
        else:
            failed.append(label)

    total_elapsed = time.perf_counter() - total_start

    # ── Save collection ──
    collection_path = "output/collect/simple_fractions.json"
    with open(collection_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_targets": len(PRIME_DENOMS) + len(EXTRA_TARGETS),
                    "succeeded": len(collection),
                    "failed": len(failed),
                    "total_elapsed_sec": total_elapsed,
                    "config": BASE_PARAMS,
                },
                "collection": collection,
                "failed": failed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(f"  DONE")
    print(f"  Succeeded: {len(collection)}  Failed: {len(failed)}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Saved to: {collection_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
