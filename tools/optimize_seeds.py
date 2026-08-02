"""
Optimize each solution in the collection: minimize nodes while keeping err=0.
- 50-gen pass: if no improvement, keep current.
- If improved, run 200-gen pass.
- Skip 4-node graphs (already minimal).
"""
import json, math, os, shutil, time
from collections import OrderedDict

from config import DEFAULT_CONFIG
from ga.fitness import evaluate_cached
from ga.core import run as ga_run


DEFAULT_CONFIG.max_edges = 40
DEFAULT_CONFIG.internal_nodes_choices = list(range(5, 35))
DEFAULT_CONFIG.internal_nodes_weights = [
    math.exp(-((x - 15) ** 2) / (2 * 2.0 ** 2)) for x in range(5, 35)
]
DEFAULT_CONFIG.stagnation_interval = 8
DEFAULT_CONFIG.stagnation_restart = 30
DEFAULT_CONFIG.stagnation_boost_ratio = 0.40
DEFAULT_CONFIG.restart_survivors = 5

OPTIM_DIR = "output/collect_optim"


def optimize_one(label: str, entry: dict) -> dict | None:
    num = entry["target"]["p"]
    den = entry["target"]["q"]
    current_nodes = entry["nodes"]
    current_edges = entry["edges_count"]

    # Skip 4-node graphs (already minimal)
    if current_nodes <= 4:
        print(f"  {label}: {current_nodes}n/{current_edges}e — already minimal, skipping")
        return entry

    out_dir = os.path.join(OPTIM_DIR, f"{num}_{den}")
    os.makedirs(out_dir, exist_ok=True)
    DEFAULT_CONFIG.history_path = os.path.join(out_dir, "ga_history.json")

    seed_edges = [(e[0], e[1]) for e in entry["edges"]]
    DEFAULT_CONFIG.seed_edges = [seed_edges]

    def run_opt(generations: int) -> dict | None:
        results = ga_run(
            target_pq=(num, den),
            pop_size=50,
            generations=generations,
            mutation_rate=0.55,
            crossover_rate=0.45,
            tournament_size=3,
            elitism_count=3,
            immigration_rate=0.18,
            mutation_weights=(0.15, 0.15, 0.10, 0.60),
            output_path=os.path.join(out_dir, "ga_top5.json"),
            mode="mixed",
        )
        best = None
        for e in (results or []):
            if e.get("error") != 0:
                continue
            edg = e["graph"]["edges"]
            et = tuple(sorted((x[0], x[1]) for x in edg))
            real_err, real_nodes = evaluate_cached(et, (num, den), mode="mixed")
            if abs(real_err) < 1e-15:
                if best is None or real_nodes < best["verified_nodes"]:
                    best = {
                        "target": {"p": num, "q": den},
                        "nodes": e["nodes"],
                        "edges_count": len(edg),
                        "edges": edg,
                        "verified_nodes": real_nodes,
                    }
        return best

    # Phase 1: 50 gens
    t0 = time.perf_counter()
    best_50 = run_opt(50)
    elapsed = time.perf_counter() - t0

    if best_50 is None:
        print(f"  {label}: optimization lost err=0! keeping original")
        return entry

    if best_50["verified_nodes"] >= current_nodes:
        print(
            f"  {label}: {current_nodes}n/{current_edges}e → {best_50['nodes']}n/{best_50['edges_count']}e "
            f"(no improvement, {elapsed:.1f}s)"
        )
        return entry if best_50["verified_nodes"] >= current_nodes else best_50

    print(
        f"  {label}: {current_nodes}n/{current_edges}e → {best_50['nodes']}n/{best_50['edges_count']}e "
        f"IMPROVED! Running 200 gens..."
    )

    # Phase 2: 200 gens
    t0 = time.perf_counter()
    DEFAULT_CONFIG.seed_edges = [[(e[0], e[1]) for e in best_50["edges"]]]
    best_200 = run_opt(200)
    elapsed2 = time.perf_counter() - t0

    if best_200 is None or best_200["verified_nodes"] > best_50["verified_nodes"]:
        print(f"  → keeping 50-gen result: {best_50['nodes']}n/{best_50['edges_count']}e")
        return best_50

    print(
        f"  → final: {best_200['nodes']}n/{best_200['edges_count']}e ({elapsed + elapsed2:.1f}s)"
    )
    return best_200


def main():
    os.makedirs(OPTIM_DIR, exist_ok=True)

    with open("output/collect_iter3/merged_fractions.json") as f:
        data = json.load(f)

    collection = data["collection"]
    print(f"Optimizing {len(collection)} solutions...\n")

    optimized = OrderedDict()
    total_t0 = time.perf_counter()
    improved_count = 0
    skipped_count = 0

    for label, entry in collection.items():
        result = optimize_one(label, entry)
        if result is None:
            result = entry
        optimized[label] = result
        if result["nodes"] < entry["nodes"]:
            improved_count += 1
        if entry["nodes"] <= 4:
            skipped_count += 1

    total_elapsed = time.perf_counter() - total_t0

    # Save
    out_path = "output/seed_graphs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Optimized seed graphs for TopoFlow — verified err=0, node-minimal",
                "total": len(optimized),
                "improved": improved_count,
                "skipped_4node": skipped_count,
                "elapsed_sec": total_elapsed,
                "collection": optimized,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total: {len(optimized)}  Improved: {improved_count}  Skipped (4n): {skipped_count}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
