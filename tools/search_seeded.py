"""
Search failed fraction targets using collection seeds as initial population.
"""
import json
import math
import os
import time
from collections import OrderedDict

from config import DEFAULT_CONFIG
from ga.fitness import evaluate_cached
from ga.core import run as ga_run


# ── Patch DEFAULT_CONFIG ──
DEFAULT_CONFIG.max_edges = 15
DEFAULT_CONFIG.internal_nodes_choices = list(range(5, 21))
DEFAULT_CONFIG.internal_nodes_weights = [
    math.exp(-((x - 10) ** 2) / (2 * 1.5 ** 2)) for x in range(5, 21)
]
DEFAULT_CONFIG.stagnation_interval = 8
DEFAULT_CONFIG.stagnation_restart = 30
DEFAULT_CONFIG.stagnation_boost_ratio = 0.40
DEFAULT_CONFIG.restart_survivors = 5


def load_seed_edges(collection_path: str) -> list[list[tuple[str, str]]]:
    with open(collection_path) as f:
        data = json.load(f)
    seeds = []
    for label, entry in data["collection"].items():
        edges = [(e[0], e[1]) for e in entry["edges"]]
        seeds.append(edges)
    return seeds


def search_one_target(num: int, den: int, seeds: list) -> dict | None:
    label = f"{num}/{den}"
    output_dir = f"output/collect_seeded/{num}_{den}"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ga_top5.json")
    DEFAULT_CONFIG.history_path = os.path.join(output_dir, "ga_history.json")
    DEFAULT_CONFIG.seed_edges = seeds

    print(f"\n{'='*50}")
    print(f"  Target: {label}  (seeds={len(seeds)})")
    print(f"{'='*50}")

    t0 = time.perf_counter()
    try:
        results = ga_run(
            target_pq=(num, den),
            pop_size=80,
            generations=150,
            mutation_rate=0.68,
            crossover_rate=0.55,
            tournament_size=3,
            elitism_count=5,
            immigration_rate=0.22,
            mutation_weights=(0.15, 0.15, 0.10, 0.60),
            output_path=out_path,
            mode="mixed",
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    if not results:
        print(f"  No results")
        return None

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
        print(f"  No verified err=0 solution")

    return best


def main():
    os.makedirs("output/collect_seeded", exist_ok=True)

    # Load seeds
    seeds = load_seed_edges("output/collect/simple_fractions.json")
    print(f"Loaded {len(seeds)} seed graphs from collection")

    # Load failed targets
    with open("output/collect/simple_fractions.json") as f:
        data = json.load(f)
    failed = data["failed"]

    # Parse failed targets back to (num, den)
    failed_targets = []
    for s in failed:
        parts = s.split("/")
        failed_targets.append((int(parts[0]), int(parts[1])))

    print(f"Failed targets to search: {len(failed_targets)}")

    new_results: dict = OrderedDict()
    still_failed: list = []
    total_start = time.perf_counter()

    for num, den in failed_targets:
        result = search_one_target(num, den, seeds)
        label = f"{num}/{den}"
        if result:
            new_results[label] = result
        else:
            still_failed.append(label)

    total_elapsed = time.perf_counter() - total_start

    # Merge with existing collection
    merged = OrderedDict()
    for k, v in data["collection"].items():
        merged[k] = v
    for k, v in new_results.items():
        merged[k] = v

    merged_path = "output/collect_seeded/merged_fractions.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total": len(merged),
                    "from_seeded": len(new_results),
                    "still_failed": len(still_failed),
                    "total_elapsed_sec": total_elapsed,
                },
                "collection": merged,
                "still_failed": still_failed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(f"  DONE")
    print(f"  Newly found: {len(new_results)}  Still failed: {len(still_failed)}")
    print(f"  Total collection: {len(merged)}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Saved to: {merged_path}")
    if new_results:
        for k, v in new_results.items():
            print(f"    {k}: {v['nodes']}n/{v['edges_count']}e")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
