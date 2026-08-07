"""Ablation: does potential-guided mutation help the GA reach exact solutions?

Runs GA with potential model ON vs OFF (same seed), compares best-error trajectory
plus err=0-oriented metrics.

Usage:
    python tools/ablate_potential.py [seeds...]
"""
import json
import os
import random
import shutil
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def run_once(tag: str, potential_model_path, seed: int = 42) -> None:
    from ga.core import run
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*60}")
    print(f"  GA ablation | {tag} | seed={seed}  potential_model={potential_model_path}")
    print(f"{'='*60}")
    run(
        target_pq=(325, 799),
        pop_size=60,
        generations=50,
        solver_workers=8,
        mode="mixed",
        surrogate_type="gnn",
        potential_model_path=potential_model_path,
    )
    src = "output/ga_history.json"
    dst = f"output/ablation_{tag}.json"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  saved history -> {dst}")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare() -> None:
    on = load("output/ablation_on.json")
    off = load("output/ablation_off.json")

    gen_on = {g["gen"]: g["best_error"] for g in on["generations"]}
    gen_off = {g["gen"]: g["best_error"] for g in off["generations"]}
    common = sorted(set(gen_on) & set(gen_off))

    wins = sum(1 for g in common if gen_on[g] < gen_off[g] - 1e-9)
    losses = sum(1 for g in common if gen_on[g] > gen_off[g] + 1e-9)

    final_on = gen_on[common[-1]]
    final_off = gen_off[common[-1]]
    best_on = min(gen_on.values())
    best_off = min(gen_off.values())

    print(f"\n  Final gen:   OFF={final_off:.6f}  ON={final_on:.6f}  delta={final_on-final_off:+.6f}")
    print(f"  Best ever:   OFF={best_off:.6f}  ON={best_on:.6f}  delta={best_on-best_off:+.6f}")
    print(f"  Gens where ON better: {wins}/{len(common)}  OFF better: {losses}/{len(common)}")
    print(f"  ON reached err<1e-3: {'YES' if best_on < 1e-3 else 'no'}  OFF: {'YES' if best_off < 1e-3 else 'no'}")
    print(f"  ON reached err<1e-4: {'YES' if best_on < 1e-4 else 'no'}  OFF: {'YES' if best_off < 1e-4 else 'no'}")
    print(f"  ON reached err<1e-5: {'YES' if best_on < 1e-5 else 'no'}  OFF: {'YES' if best_off < 1e-5 else 'no'}")
    print(f"  ON reached err==0:   {'YES' if best_on == 0.0 else 'no'}  OFF: {'YES' if best_off == 0.0 else 'no'}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 123, 999, 742]
    for seed in seeds:
        run_once("off", None, seed=seed)
        run_once("on", "output/potential_net.pt", seed=seed)
        compare()
