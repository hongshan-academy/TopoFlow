"""Profiles every major pipeline stage with warmup + timed runs.

Usage:  uv run tools/profile.py   (takes ~2-3 min)
Output: prints a ranked hotspot table to stdout.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import statistics
import time
import gc
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np

from config import DEFAULT_CONFIG as _cfg
from ga.fitness import evaluate_cached
from ga.generation import generate_strict_graph
from ga.mutation import (
    mutate_partial_delete,
    mutate_add_subgraph,
    mutate_replace_subgraph,
    mutate_reverse_edge,
)
from ga.crossover import crossover_subgraph_exchange
from ga.utils import find_1in_1out_subgraph
from graph import Graph, Edge
from simulator import TopoFlowSimulator
from solver import solve

# ── helpers ──────────────────────────────────────────────────────────────────

def _tick() -> float:
    return time.perf_counter()


def _elapsed_since(start: float) -> float:
    return time.perf_counter() - start


def _med(vals: List[float]) -> float:
    n = len(vals)
    if n == 0:
        return float("nan")
    return (vals[n // 2] + vals[(n - 1) // 2]) / 2


def _time_it(
    fn: Callable[[], Any],
    warmup: int = 2,
    repeats: int = 5,
    gc_between: bool = False,
):
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(repeats):
        if gc_between:
            gc.collect()
        t0 = _tick()
        fn()
        times.append(_elapsed_since(t0))
    return {
        "min_ms": min(times) * 1000,
        "med_ms": _med(times) * 1000,
        "mean_ms": statistics.mean(times) * 1000,
        "max_ms": max(times) * 1000,
        "raw_s": times,
    }


# ── graph pool ───────────────────────────────────────────────────────────────

def _make_graph_pool(seeds: int = 20) -> List[Graph]:
    """Generate a diverse set of valid graphs for profiling."""
    random.seed(42)
    graphs: List[Graph] = []
    seen: Set[Tuple[Edge, ...]] = set()
    attempts = 0
    while len(graphs) < seeds and attempts < 500:
        n_internal = random.choices([10, 20, 30, 40, 50], weights=[1, 2, 3, 2, 1], k=1)[0]
        g = generate_strict_graph(n_internal)
        key = tuple(sorted(g.edges))
        if key not in seen and g.is_valid(strict=True) and len(g.edges) >= 3:
            seen.add(key)
            graphs.append(g)
        attempts += 1
    return graphs


def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_hotspots(results: List[Dict[str, Any]]) -> None:
    results.sort(key=lambda r: r["med_ms"], reverse=True)
    print(f"\n{'=' * 80}")
    print(f"  RANKED HOTSPOTS  (by median ms, most expensive first)")
    print(f"{'=' * 80}")
    print(f"  {'#':>3}  {'Stage':<40} {'med.ms':>9} {'mean.ms':>9} {'min.ms':>9} {'max.ms':>9}")
    print(f"  {'─' * 3}  {'─' * 40} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}")
    for i, r in enumerate(results):
        print(f"  {i+1:>3}  {r['name']:<40} {r['med_ms']:>8.2f} {r['mean_ms']:>8.2f} "
              f"{r['min_ms']:>8.2f} {r['max_ms']:>8.2f}")
    print(f"{'=' * 80}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("  TopoFlow — Full Pipeline Profile")
    print("=" * 80)

    print_section("0. Building graph pool...")
    all_graphs = _make_graph_pool(seeds=20)
    print(f"  generated {len(all_graphs)} diverse graphs  |  "
          f"edges range: {min(len(g.edges) for g in all_graphs)}-{max(len(g.edges) for g in all_graphs)}")

    hotspots: List[Dict[str, Any]] = []

    # ── 1. Graph generation ──────────────────────────────────────────────────
    print_section("1. generate_strict_graph")
    for label, n_int in [("small  (10)", 10), ("medium (30)", 30), ("large  (50)", 50)]:
        def _gen():
            random.seed(42)
            return generate_strict_graph(n_int)

        t = _time_it(_gen, warmup=3, repeats=10, gc_between=True)
        hotspots.append({
            "name": f"generate_strict_graph  n_int={n_int}",
            **t,
        })
        print(f"  {label}:  med={t['med_ms']:.2f}ms  mean={t['mean_ms']:.2f}ms  "
              f"min={t['min_ms']:.2f}ms  max={t['max_ms']:.2f}ms")

    # ── 2. Graph.copy ────────────────────────────────────────────────────────
    print_section("2. Graph.copy")
    for g in all_graphs[:5]:
        t = _time_it(lambda g=g: g.copy(), warmup=5, repeats=20, gc_between=True)
        hotspots.append({
            "name": f"Graph.copy  (nodes={len(g.nodes)} edges={len(g.edges)})",
            **t,
        })
    avgs_copy = [h["med_ms"] for h in hotspots if h["name"].startswith("Graph.copy")]
    if avgs_copy:
        print(f"  average med: {statistics.mean(avgs_copy):.2f}ms  |  range: {min(avgs_copy):.2f}-{max(avgs_copy):.2f}ms")

    # ── 3. find_1in_1out_subgraph ────────────────────────────────────────────
    print_section("3. find_1in_1out_subgraph")
    for g in all_graphs[:5]:
        t = _time_it(lambda g=g: find_1in_1out_subgraph(g), warmup=3, repeats=5, gc_between=True)
        hotspots.append({
            "name": f"find_1in_1out_subgraph  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  med={t['med_ms']:.3f}ms  "
              f"mean={t['mean_ms']:.3f}ms  max={t['max_ms']:.3f}ms")

    # ── 4. MILP Solver ───────────────────────────────────────────────────────
    print_section("4. MILP solver  (solve, threads=1)")
    milp_graphs = [g for g in all_graphs if len(g.edges) <= 60]
    if not milp_graphs:
        milp_graphs = all_graphs[:3]
    for g in milp_graphs[:5]:
        def _sol(g=g):
            return solve(g, threads=1)

        t = _time_it(_sol, warmup=1, repeats=3, gc_between=True)
        hotspots.append({
            "name": f"MILP solve  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        status = _sol().status
        import pulp
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3} [{pulp.LpStatus[status]}]:  "
              f"med={t['med_ms']:.1f}ms  mean={t['mean_ms']:.1f}ms  max={t['max_ms']:.1f}ms")

    # ── 5. Simulator ─────────────────────────────────────────────────────────
    print_section("5. Simulator  (run_until_cycle, max_frames=100000, threads=1)")
    for g in all_graphs[:5]:
        def _sim(g=g):
            sim = TopoFlowSimulator(g)
            return sim.run_until_cycle(max_frames=100000)

        t = _time_it(_sim, warmup=1, repeats=3, gc_between=True)
        converged = _sim().get("converged", False)
        hotspots.append({
            "name": f"Simulator run  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        info = _sim()
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3} "
              f"[conv={converged} frames={info.get('total_frames',0)}]:  "
              f"med={t['med_ms']:.2f}ms  mean={t['mean_ms']:.2f}ms  max={t['max_ms']:.2f}ms")

    # ── 6. Step_once profiling ───────────────────────────────────────────────
    print_section("6. Simulator step_once  (single frame)")
    for g in all_graphs[:3]:
        sim = TopoFlowSimulator(g)

        def _step(sim=sim):
            return sim.step_once()

        sim.reset()
        t = _time_it(_step, warmup=5, repeats=20, gc_between=False)
        hotspots.append({
            "name": f"Sim step_once  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  "
              f"med={t['med_ms']:.5f}ms  mean={t['mean_ms']:.5f}ms  max={t['max_ms']:.5f}ms")

    # ── 7. Simulator serialize_state ─────────────────────────────────────────
    print_section("7. Simulator serialize_state")
    for g in all_graphs[:3]:
        sim = TopoFlowSimulator(g)

        def _ser(sim=sim):
            return sim.serialize_state()

        t = _time_it(_ser, warmup=10, repeats=30, gc_between=False)
        hotspots.append({
            "name": f"Sim serialize_state  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  "
              f"med={t['med_ms']:.5f}ms  mean={t['mean_ms']:.5f}ms  max={t['max_ms']:.5f}ms")

    # ── 8. Mutation operators ────────────────────────────────────────────────
    print_section("8. Mutation operators")
    mut_fns = [
        ("mutate_partial_delete", mutate_partial_delete),
        ("mutate_add_subgraph", mutate_add_subgraph),
        ("mutate_replace_subgraph", mutate_replace_subgraph),
        ("mutate_reverse_edge", mutate_reverse_edge),
    ]
    for name, fn in mut_fns:
        times_all: List[float] = []
        for g in all_graphs[:8]:
            def _mut(g=g, fn=fn):
                return fn(g)

            t = _time_it(_mut, warmup=1, repeats=5, gc_between=True)
            times_all.extend(t["raw_s"])
        if times_all:
            times_ms = [x * 1000 for x in times_all]
            n = len(times_ms)
            med = (times_ms[n // 2] + times_ms[(n - 1) // 2]) / 2
            hotspots.append({
                "name": f"{name}  (avg across graphs)",
                "med_ms": med,
                "mean_ms": statistics.mean(times_ms),
                "min_ms": min(times_ms),
                "max_ms": max(times_ms),
                "raw_s": times_all,
                "ncalls": len(times_all),
            })
            print(f"  {name:<30}:  med={med:.3f}ms  mean={statistics.mean(times_ms):.3f}ms  "
                  f"min={min(times_ms):.3f}ms  max={max(times_ms):.3f}ms  (n={len(times_all)})")

    # ── 9. Crossover ─────────────────────────────────────────────────────────
    print_section("9. Crossover  (crossover_subgraph_exchange)")
    for i in range(0, min(6, len(all_graphs)), 2):
        g1 = all_graphs[i]
        g2 = all_graphs[i + 1]

        def _cx(g1=g1, g2=g2):
            return crossover_subgraph_exchange(g1, g2)

        t = _time_it(_cx, warmup=2, repeats=5, gc_between=True)
        hotspots.append({
            "name": f"crossover  (n1={len(g1.nodes)} e1={len(g1.edges)} n2={len(g2.nodes)} e2={len(g2.edges)})",
            **t,
        })
        print(f"  g1({len(g1.nodes)}n/{len(g1.edges)}e) x g2({len(g2.nodes)}n/{len(g2.edges)}e):  "
              f"med={t['med_ms']:.3f}ms  mean={t['mean_ms']:.3f}ms")

    # ── 10. RF Surrogate — feature extraction ────────────────────────────────
    print_section("10. RF Surrogate — extract_features")
    from rf_surrogate.features import extract_features
    for g in all_graphs[:5]:
        et = tuple(sorted(g.edges))

        def _ext(et=et):
            return extract_features(et)

        t = _time_it(_ext, warmup=5, repeats=30, gc_between=True)
        hotspots.append({
            "name": f"extract_features  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  "
              f"med={t['med_ms']:.5f}ms  mean={t['mean_ms']:.5f}ms  max={t['max_ms']:.5f}ms")

    # ── 11. RF Surrogate — fit + predict ─────────────────────────────────────
    print_section("11. RF Surrogate — fit & predict")
    from rf_surrogate.archive import SurrogateArchive
    from rf_surrogate.model import SurrogateRF
    archive = SurrogateArchive()
    for g in all_graphs:
        et = tuple(sorted(g.edges))
        fit_val = random.uniform(0.0, 1.0)
        archive.add(et, fit_val)

    X, y = archive.get_data()
    print(f"  archive: {archive.size()} samples, X.shape={X.shape}")

    def _rf_fit():
        m = SurrogateRF(n_estimators=150, max_depth=15)
        m.fit(X, y)
        return m

    t_fit = _time_it(_rf_fit, warmup=1, repeats=5, gc_between=True)
    hotspots.append({
        "name": "RF fit  (n_est=150, n_samples=20)",
        **t_fit,
    })
    print(f"  RF.fit({X.shape}):  med={t_fit['med_ms']:.1f}ms  mean={t_fit['mean_ms']:.1f}ms")

    model = _rf_fit()
    def _rf_pred(m=model):
        return m.predict(X)
    t_pred = _time_it(_rf_pred, warmup=3, repeats=10, gc_between=True)
    hotspots.append({
        "name": "RF predict  (n_samples=20)",
        **t_pred,
    })
    print(f"  RF.predict({X.shape}):  med={t_pred['med_ms']:.1f}ms  mean={t_pred['mean_ms']:.1f}ms")

    # ── 12. archive.get_data ─────────────────────────────────────────────────
    print_section("12. SurrogateArchive.get_data")
    archive_full = SurrogateArchive()
    for g in all_graphs:
        for _ in range(50):  # simulate fuller archive
            et = tuple(sorted(g.edges))
            archive_full.add(et, random.uniform(0.0, 1.0))

    def _get_data(a=archive_full):
        return a.get_data()

    t_arch = _time_it(_get_data, warmup=2, repeats=5, gc_between=True)
    hotspots.append({
        "name": f"archive.get_data  (n={archive_full.size()})",
        **t_arch,
    })
    print(f"  archive.get_data (n={archive_full.size()}):  "
          f"med={t_arch['med_ms']:.1f}ms  mean={t_arch['mean_ms']:.1f}ms")

    # ── 13. evaluate_cached ──────────────────────────────────────────────────
    print_section("13. evaluate_cached  (full fitness pipeline)")
    modes = ["simulation", "MILP"]
    for mode in modes:
        evals = []
        for g in all_graphs[:5]:
            et = tuple(sorted(g.edges))

            def _ev(et=et, mode=mode):
                return evaluate_cached(et, (325, 799), threads=1, max_denominator=10000, mode=mode)

            try:
                t = _time_it(_ev, warmup=1, repeats=3, gc_between=True)
                evals.append({"med_ms": t["med_ms"], "mean_ms": t["mean_ms"]})
            except Exception:
                pass

        if evals:
            meds = [e["med_ms"] for e in evals]
            means = [e["mean_ms"] for e in evals]
            label = f"evaluate_cached  mode={mode}  (avg across graphs)"
            hotspots.append({
                "name": label,
                "med_ms": statistics.mean(meds),
                "mean_ms": statistics.mean(means),
                "min_ms": min(meds),
                "max_ms": max(meds),
            })
            print(f"  {mode:<12}:  avg_med={statistics.mean(meds):.1f}ms  avg_mean={statistics.mean(means):.1f}ms  "
                  f"range=[{min(meds):.1f}, {max(meds):.1f}]ms")

    # ── 14. GA operators overhead ────────────────────────────────────────────
    print_section("14. Per-generation GA overhead  (without solver)")
    # Simulate a generation to measure bookkeeping
    from deap import base, creator, tools as deap_tools
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", tuple, fitness=creator.FitnessMin)

    pop_sample = []
    for i in range(100):
        g = all_graphs[i % len(all_graphs)]
        ind = creator.Individual(tuple(sorted(g.edges)))
        ind.fitness.values = (random.uniform(0.0, 0.5), len(g.nodes))
        pop_sample.append(ind)

    toolbox = base.Toolbox()
    toolbox.register("select", deap_tools.selTournament, tournsize=2)

    def _ga_overhead():
        selected = toolbox.select(pop_sample, 50)
        selected = [toolbox.clone(ind) for ind in selected]
        for ind in selected:
            g = Graph.from_edges(list(ind))
            _ = len(g.edges)
        sorted(pop_sample, key=lambda ind: ind.fitness.values)

    t_ga = _time_it(_ga_overhead, warmup=3, repeats=10, gc_between=True)
    hotspots.append({
        "name": "GA per-gen overhead  (select+clone+Graph.from_edges+sort, pop=100)",
        **t_ga,
    })
    print(f"  pop=100:  med={t_ga['med_ms']:.2f}ms  mean={t_ga['mean_ms']:.2f}ms  "
          f"min={t_ga['min_ms']:.2f}ms  max={t_ga['max_ms']:.2f}ms")

    # ── 15. Graph.from_edges ─────────────────────────────────────────────────
    print_section("15. Graph.from_edges")
    for g in all_graphs[:5]:
        et = list(g.edges)
        t = _time_it(lambda et=et: Graph.from_edges(et), warmup=5, repeats=30, gc_between=True)
        hotspots.append({
            "name": f"Graph.from_edges  (n={len(g.nodes)} e={len(g.edges)})",
            **t,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  "
              f"med={t['med_ms']:.5f}ms  mean={t['mean_ms']:.5f}ms max={t['max_ms']:.5f}ms")

    # ── 16. Simulator mini-bench: frame cost vs graph size ───────────────────
    print_section("16. Simulator frame-cost breakdown")
    for g in all_graphs[:3]:
        sim = TopoFlowSimulator(g)
        sim.reset()
        n_frames = 1000
        t0 = _tick()
        for _ in range(n_frames):
            sim.step_once()
        total = _elapsed_since(t0)
        per_frame = total / n_frames * 1000
        hotspots.append({
            "name": f"Sim 1000-frame avg  (n={len(g.nodes)} e={len(g.edges)})",
            "med_ms": per_frame,
            "mean_ms": per_frame,
            "min_ms": per_frame,
            "max_ms": per_frame,
        })
        print(f"  nodes={len(g.nodes):>3} edges={len(g.edges):>3}:  "
              f"{total*1000:.1f}ms total / {n_frames} frames = {per_frame:.5f}ms/frame")

    # ── print hotspot ranking ────────────────────────────────────────────────
    print_hotspots(hotspots)


if __name__ == "__main__":
    main()
