"""
Deep seeded search for remaining failed fraction targets.
max_edges=40, all 24 known solutions as seeds, gen=300.
"""
import json
import math
import os
import time
from collections import OrderedDict

from config import DEFAULT_CONFIG
from ga.fitness import evaluate_cached
from ga.core import run as ga_run


DEFAULT_CONFIG.max_edges = 40
DEFAULT_CONFIG.internal_nodes_choices = list(range(5, 35))
DEFAULT_CONFIG.internal_nodes_weights = [
    math.exp(-((x - 15) ** 2) / (2 * 2.0 ** 2)) for x in range(5, 35)
]
DEFAULT_CONFIG.stagnation_interval = 10
DEFAULT_CONFIG.stagnation_restart = 60
DEFAULT_CONFIG.stagnation_boost_ratio = 0.40
DEFAULT_CONFIG.restart_survivors = 8


def load_seeds(path: str) -> list[list[tuple[str, str]]]:
    with open(path) as f:
        data = json.load(f)
    seeds = []
    for entry in data["collection"].values():
        edges = [(e[0], e[1]) for e in entry["edges"]]
        seeds.append(edges)
    return seeds


def search_one(num: int, den: int, seeds: list) -> dict | None:
    label = f"{num}/{den}"
    out_dir = f"output/collect_iter3/{num}_{den}"
    os.makedirs(out_dir, exist_ok=True)
    DEFAULT_CONFIG.history_path = os.path.join(out_dir, "ga_history.json")
    DEFAULT_CONFIG.seed_edges = seeds

    print(f"\n{'='*50}")
    print(f"  {label}  (seeds={len(seeds)})")
    print(f"{'='*50}")

    t0 = time.perf_counter()
    try:
        results = ga_run(
            target_pq=(num, den),
            pop_size=80,
            generations=600,
            mutation_rate=0.68,
            crossover_rate=0.55,
            tournament_size=3,
            elitism_count=5,
            immigration_rate=0.22,
            mutation_weights=(0.15, 0.15, 0.10, 0.60),
            output_path=os.path.join(out_dir, "ga_top5.json"),
            mode="mixed",
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    if not results:
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
        print(f"  VERIFIED: {best['nodes']}n/{best['edges_count']}e")
    else:
        print(f"  No verified err=0")
    return best


def main():
    os.makedirs("output/collect_iter3", exist_ok=True)

    seeds = load_seeds("output/collect_iter2/merged_fractions.json")
    print(f"Loaded {len(seeds)} seed graphs")

    with open("output/collect_iter2/merged_fractions.json") as f:
        data = json.load(f)

    still_failed = data.get("still_failed", [])
    targets = []
    for s in still_failed:
        parts = s.split("/")
        targets.append((int(parts[0]), int(parts[1])))
    print(f"Remaining targets: {len(targets)}")

    new_results: dict = OrderedDict()
    new_failed: list = []

    for num, den in targets:
        result = search_one(num, den, seeds)
        label = f"{num}/{den}"
        if result:
            new_results[label] = result
            seeds.append([(e[0], e[1]) for e in result["edges"]])
        else:
            new_failed.append(label)

    # Merge
    merged = OrderedDict()
    for k, v in data["collection"].items():
        merged[k] = v
    for k, v in new_results.items():
        merged[k] = v

    out_path = "output/collect_iter3/merged_fractions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total": len(merged),
                    "new": len(new_results),
                    "still_failed": len(new_failed),
                },
                "collection": merged,
                "still_failed": new_failed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'='*60}")
    print(f"  New: {len(new_results)}  Still failed: {len(new_failed)}")
    print(f"  Total: {len(merged)}")
    if new_results:
        for k, v in new_results.items():
            print(f"    {k}: {v['nodes']}n/{v['edges_count']}e")
    print(f"  Saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
