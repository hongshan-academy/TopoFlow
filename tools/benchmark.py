"""Benchmark MILP vs simulation solver speed by edge count.

ProcessPoolExecutor (16 workers), runs ~12 minutes.
Tracks MILP success/fail and SIM convergence separately.
Only converged results used for timing comparison.
"""

import gc
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_WORKERS = 16
_RUNTIME_S = 720
_MAX_SIM_FRAMES = 100000
_EDGE_BUCKET = 5
_MIN_PER_BUCKET = 10
_SEED = 42
_MAX_EDGES = 75


# ── Result types ────────────────────────────────────────────────────────────

@dataclass
class Sample:
    nodes: int
    edges: int
    milp_ok: bool      # MILP solved to optimality
    milp_ms: float      # ms, valid only if milp_ok
    sim_ok: bool         # simulation converged (found cycle)
    sim_ms: float        # ms, valid only if sim_ok
    sim_frames: int      # total frames run


@dataclass
class BucketStats:
    edge_low: int
    edge_high: int
    n_total: int         # all samples in this bucket
    n_milp_ok: int       # MILP optimal
    n_sim_ok: int        # simulation converged
    n_both_ok: int       # both succeeded
    milp_mean_ms: float
    milp_med_ms: float
    sim_mean_ms: float
    sim_med_ms: float
    sim_mean_frames: float
    winner: str          # 'MILP' | 'simulation' | 'tie'
    samples: List[Sample] = field(default_factory=list)


# ── Worker ──────────────────────────────────────────────────────────────────

def _do_one(n_internal: int, seed: int) -> Optional[Tuple[int, int, int, float, int, float, int]]:
    """Runs both solvers and returns detailed success/failure + timing info."""
    random.seed(seed)
    from ga.generation import generate_strict_graph
    from solver import solve
    from simulator import TopoFlowSimulator
    import pulp as _pulp

    try:
        graph = generate_strict_graph(n_internal)
    except Exception:
        gc.collect()
        return None
    if len(graph.edges) > _MAX_EDGES or len(graph.nodes) < 3:
        gc.collect()
        return None
    if not graph.is_valid(strict=True):
        gc.collect()
        return None

    nodes = len(graph.nodes)
    edges = len(graph.edges)

    # ── MILP ──
    milp_ok = False
    milp_ms = -1.0
    try:
        t0 = time.perf_counter()
        result = solve(graph, threads=1)
        milp_ms = (time.perf_counter() - t0) * 1000
        milp_ok = (result.status == _pulp.constants.LpStatusOptimal)
    except Exception:
        milp_ok = False

    # ── Simulation ──
    sim_ok = False
    sim_ms = -1.0
    sim_frames = 0
    try:
        t0 = time.perf_counter()
        sim_obj = TopoFlowSimulator(graph)
        cycle_info = sim_obj.run_until_cycle(max_frames=_MAX_SIM_FRAMES)
        sim_ms = (time.perf_counter() - t0) * 1000
        sim_frames = cycle_info.get('total_frames', 0)
        sim_ok = cycle_info.get('converged', False)
    except Exception:
        pass

    gc.collect()

    return (nodes, edges, int(milp_ok), milp_ms, int(sim_ok), sim_ms, sim_frames)


# ── Bucket helpers ──────────────────────────────────────────────────────────

def _bkt_range(edges: int) -> Tuple[int, int]:
    lo = (edges // _EDGE_BUCKET) * _EDGE_BUCKET
    return lo, lo + _EDGE_BUCKET - 1


def _bkt_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo:>3d}-{hi:>3d}"


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("  TopoFlow Solver Benchmark  (by edge count)")
    print(f"  Workers: {_WORKERS}  |  Runtime: ~{_RUNTIME_S}s  |  Sim max frames: {_MAX_SIM_FRAMES}")
    print(f"  Bucket: {_EDGE_BUCKET} edges  |  Max edges: {_MAX_EDGES}")
    print("=" * 80)

    rng = random.Random(_SEED)
    n_int_range = list(range(3, 49))
    seed_base = _SEED * 1000000
    task_iter = 0

    raw: List[Tuple[int, int, int, float, int, float, int]] = []
    n_timeout = 0
    n_submitted = 0
    t_deadline = time.perf_counter() + _RUNTIME_S

    pbar = tqdm.tqdm(total=None, desc="  Benchmark", unit="t", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {}

        def _fill(cnt: int = 1) -> None:
            nonlocal task_iter, n_submitted
            for _ in range(cnt):
                if time.perf_counter() >= t_deadline:
                    return
                n_int = rng.choice(n_int_range)
                fut = pool.submit(_do_one, n_int, seed_base + task_iter)
                futures[fut] = time.perf_counter()
                task_iter += 1
                n_submitted += 1

        _fill(_WORKERS * 4)

        while time.perf_counter() < t_deadline:
            remaining = max(0, t_deadline - time.perf_counter())
            pbar.set_postfix_str(
                f"ok={len(raw)} t/o={n_timeout} sub={n_submitted} "
                f"pend={len(futures)} {remaining:.0f}s"
            )

            try:
                done_iter = as_completed(futures, timeout=min(5, max(1, remaining)))
                for fut in done_iter:
                    dt = time.perf_counter() - futures.pop(fut, float('inf'))
                    try:
                        r = fut.result(timeout=0)
                    except Exception:
                        r = None
                    if r is not None:
                        raw.append(r)
                    _fill()
                    pbar.update(1)
                    if time.perf_counter() >= t_deadline:
                        break
            except TimeoutError:
                pass
            except Exception:
                pass

            if not futures and time.perf_counter() >= t_deadline:
                break

        for fut in list(futures):
            fut.cancel()

    pbar.close()

    elapsed = time.perf_counter() - (t_deadline - _RUNTIME_S)
    samples = [Sample(nodes=r[0], edges=r[1], milp_ok=bool(r[2]), milp_ms=r[3], sim_ok=bool(r[4]), sim_ms=r[5], sim_frames=r[6]) for r in raw]
    n_milp_fail = sum(1 for s in samples if not s.milp_ok)
    n_sim_fail = sum(1 for s in samples if not s.sim_ok)
    print(f"\n  {elapsed:.0f}s  |  {len(samples)} total  |  {n_timeout} timeout  |  "
          f"MILP fail={n_milp_fail}  SIM fail={n_sim_fail}\n")

    if not samples:
        print("  No valid samples.")
        return

    # ── bucket ──────────────────────────────────────────────────────────
    buckets: Dict[Tuple[int, int], List[Sample]] = {}
    for s in samples:
        buckets.setdefault(_bkt_range(s.edges), []).append(s)

    stats_list: List[BucketStats] = []
    for (lo, hi) in sorted(buckets):
        bucket = buckets[(lo, hi)]
        if len(bucket) < _MIN_PER_BUCKET:
            continue

        n_milp = [s for s in bucket if s.milp_ok]
        n_sim  = [s for s in bucket if s.sim_ok]
        n_both = [s for s in bucket if s.milp_ok and s.sim_ok]

        milp_pts = sorted(s.milp_ms for s in n_milp)
        sim_pts  = sorted(s.sim_ms for s in n_sim)
        sim_fr   = [s.sim_frames for s in n_sim]

        def _med(vals: List[float]) -> float:
            n = len(vals)
            if n == 0:
                return float('nan')
            return (vals[n // 2] + vals[(n - 1) // 2]) / 2

        milp_mean = statistics.mean(milp_pts) if milp_pts else float('nan')
        milp_med  = _med(milp_pts)
        sim_mean  = statistics.mean(sim_pts) if sim_pts else float('nan')
        sim_med   = _med(sim_pts)
        sim_fmean = statistics.mean(sim_fr) if sim_fr else float('nan')

        # Winner: compare on the subset where BOTH succeeded
        both_milp = sorted(s.milp_ms for s in n_both)
        both_sim  = sorted(s.sim_ms  for s in n_both)
        if len(both_milp) >= 3:
            bm = statistics.mean(both_milp)
            bs = statistics.mean(both_sim)
            winner = 'MILP' if bm <= bs else 'simulation'
        elif milp_pts and sim_pts:
            winner = 'MILP' if milp_mean <= sim_mean else 'simulation'
        else:
            winner = 'MILP' if milp_pts else 'simulation'

        stats_list.append(BucketStats(
            edge_low=lo, edge_high=hi,
            n_total=len(bucket),
            n_milp_ok=len(n_milp),
            n_sim_ok=len(n_sim),
            n_both_ok=len(n_both),
            milp_mean_ms=milp_mean, milp_med_ms=milp_med,
            sim_mean_ms=sim_mean, sim_med_ms=sim_med,
            sim_mean_frames=sim_fmean,
            winner=winner,
            samples=bucket,
        ))

    if not stats_list:
        print("  No buckets with enough samples.")
        return

    # ── print table ──────────────────────────────────────────────────────
    print(f"  {'Edges':>10}  {'N':>4}  {'MILP ok':>7}  {'SIM ok':>7}  {'Both':>5}  "
          f"{'MILP mean':>10}  MILP med     {'SIM mean':>9}  SIM med      "
          f"{'SIM fr':>7}  Winner")
    print("  " + "-" * 98)
    for st in stats_list:
        lbl = _bkt_label(st.edge_low, st.edge_high)
        w   = "MILP" if st.winner == 'MILP' else "SIM "
        print(f"  {lbl:>10}  {st.n_total:>4}  {st.n_milp_ok:>7}  {st.n_sim_ok:>7}  {st.n_both_ok:>5}  "
              f"{st.milp_mean_ms:>8.1f}ms  {st.milp_med_ms:>8.1f}ms  "
              f"{st.sim_mean_ms:>8.1f}ms  {st.sim_med_ms:>8.1f}ms  "
              f"{st.sim_mean_frames:>6.0f}  {w}")

    # ── summary: compute threshold from valid both-ok samples ────────────
    first_winner = stats_list[0].winner if stats_list else 'simulation'
    threshold: Optional[int] = None
    stable = 0
    for i in range(1, len(stats_list)):
        cur = stats_list[i]
        if cur.winner != first_winner:
            stable += 1
            if stable >= 2 and threshold is None:
                threshold = cur.edge_low
        else:
            stable = max(0, stable - 1)

    print(f"\n{'=' * 80}")
    print(f"  MILP total OK: {sum(s.n_milp_ok for s in stats_list)}  |  "
          f"SIM total OK: {sum(s.n_sim_ok for s in stats_list)}  |  "
          f"Both OK: {sum(s.n_both_ok for s in stats_list)}")
    if threshold is not None:
        print(f"  Crossover at {threshold} edges")
        if first_winner == 'simulation':
            print(f"  edges <= {threshold} → simulation  |  edges > {threshold} → MILP")
        else:
            print(f"  edges <= {threshold} → MILP       |  edges > {threshold} → simulation")
    else:
        wn = "simulation" if first_winner == 'simulation' else "MILP"
        print(f"  No crossover — {wn} always faster")
        threshold = 0 if first_winner == 'simulation' else 999999
    print(f"  mixed_edge_threshold = {threshold}")
    print(f"{'=' * 80}")

    os.makedirs("output", exist_ok=True)
    with open("output/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "runtime_s": round(elapsed),
            "n_submitted": n_submitted, "n_total": len(samples),
            "n_timeout": n_timeout,
            "n_milp_fail": n_milp_fail, "n_sim_fail": n_sim_fail,
            "threshold": threshold,
            "convention": f"edges <= {threshold} => {first_winner}",
            "buckets": [
                {"edge_low": s.edge_low, "edge_high": s.edge_high,
                 "n_total": s.n_total, "n_milp_ok": s.n_milp_ok,
                 "n_sim_ok": s.n_sim_ok, "n_both_ok": s.n_both_ok,
                 "milp_mean_ms": round(s.milp_mean_ms, 2) if s.milp_mean_ms == s.milp_mean_ms else None,
                 "milp_med_ms": round(s.milp_med_ms, 2) if s.milp_med_ms == s.milp_med_ms else None,
                 "sim_mean_ms": round(s.sim_mean_ms, 2) if s.sim_mean_ms == s.sim_mean_ms else None,
                 "sim_med_ms": round(s.sim_med_ms, 2) if s.sim_med_ms == s.sim_med_ms else None,
                 "sim_mean_frames": round(s.sim_mean_frames, 0) if s.sim_mean_frames == s.sim_mean_frames else None,
                 "winner": s.winner}
                for s in stats_list
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → output/benchmark_results.json")


if __name__ == "__main__":
    main()
