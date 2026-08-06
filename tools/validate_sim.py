"""Validate simulation accuracy against MILP ground truth on random strict graphs.

Compares the source-edge flow ratio produced by the discrete-event simulator
with the exact MILP solution. Measures convergence rate, mean absolute error,
and agreement rate, broken down by edge count.

Uses ProcessPoolExecutor for parallelism. Targets ~9 minutes runtime.
"""

import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_WORKERS = 16
_RUNTIME_S = 420
_MAX_SIM_FRAMES = 100000
_EDGE_BUCKET = 10
_MIN_PER_BUCKET = 3
_SEED = 42
_MAX_EDGES = 100
_N_INTERNAL_RANGE = list(range(3, 45))

_EPSILON = 1e-15


@dataclass
class Sample:
    nodes: int
    edges: int
    milp_ok: bool
    milp_flow: float
    sim_flow: float
    sim_converged: bool


@dataclass
class BucketStats:
    edge_low: int
    edge_high: int
    n_total: int
    n_milp_fail: int
    n_converged: int
    converge_rate: float
    mae: float          # mean abs error (converged only)
    max_error: float
    agreement_rate: float  # fraction with error < EPSILON


def _bkt_range(e: int) -> Tuple[int, int]:
    lo = (e // _EDGE_BUCKET) * _EDGE_BUCKET
    return lo, lo + _EDGE_BUCKET - 1


def _bkt_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _extract_source_flow_worker(graph, result):
    source = next(iter(graph.sources))
    source_edge = graph.out_edges[source][0]
    for edge_result in result.edges:
        if edge_result.source == source and edge_result.target == source_edge[1]:
            return edge_result.flow
    return 0.0


def _do_one(n_internal: int, seed_val: int) -> Optional[Tuple]:
    import gc
    import pulp as _pulp
    from ga.generation import generate_strict_graph
    from solver import solve
    from simulator import simulate

    random.seed(seed_val)

    try:
        graph = generate_strict_graph(n_internal)
    except Exception:
        gc.collect()
        return None
    if len(graph.edges) > _MAX_EDGES or len(graph.edges) < 2:
        gc.collect()
        return None
    if not graph.is_valid(strict=True):
        gc.collect()
        return None

    nodes = len(graph.nodes)
    edges = len(graph.edges)

    milp_ok = False
    milp_flow = 0.0
    try:
        result_milp = solve(graph, threads=1)
        milp_ok = result_milp.status == _pulp.constants.LpStatusOptimal
        if milp_ok:
            milp_flow = _extract_source_flow_worker(graph, result_milp)
    except Exception:
        pass

    sim_converged = False
    sim_flow = 0.0
    try:
        sim_result = simulate(graph, max_frames=_MAX_SIM_FRAMES)
        sim_converged = sim_result.converged
        sim_flow = _extract_source_flow_worker(graph, sim_result)
    except Exception:
        pass

    gc.collect()
    return (nodes, edges, int(milp_ok), milp_flow, sim_flow, int(sim_converged))


def main() -> None:
    print("=" * 78)
    print("  TopoFlow Simulation Accuracy Validation")
    print(f"  Workers: {_WORKERS}  |  Runtime target: ~{_RUNTIME_S}s  |  Sim max frames: {_MAX_SIM_FRAMES}")
    print(f"  Bucket size: {_EDGE_BUCKET} edges  |  Min per bucket: {_MIN_PER_BUCKET}")
    print(f"  Agreement threshold: {_EPSILON}")
    print("=" * 78)

    rng = random.Random(_SEED)
    seed_base = _SEED * 1000000
    task_iter = 0

    raw: List[Tuple] = []
    n_submitted = 0
    n_generation_fail = 0
    t_deadline = time.perf_counter() + _RUNTIME_S

    pbar = tqdm.tqdm(total=None, desc="  Validate", unit="t", dynamic_ncols=True)

    pool = ProcessPoolExecutor(max_workers=_WORKERS)
    futures: Dict = {}
    try:
        def _fill(cnt: int = 1) -> None:
            nonlocal task_iter, n_submitted
            for _ in range(cnt):
                if time.perf_counter() >= t_deadline:
                    return
                n_int = rng.choice(_N_INTERNAL_RANGE)
                fut = pool.submit(_do_one, n_int, seed_base + task_iter)
                futures[fut] = True
                task_iter += 1
                n_submitted += 1

        _fill(_WORKERS * 4)

        while time.perf_counter() < t_deadline:
            remaining = max(0, t_deadline - time.perf_counter())
            n_milp_ok = len(raw)
            pbar.set_postfix_str(
                f"ok={n_milp_ok} sub={n_submitted} pend={len(futures)} {remaining:.0f}s"
            )

            try:
                done_iter = as_completed(futures, timeout=min(5, max(1, remaining)))
                for fut in done_iter:
                    futures.pop(fut, None)
                    try:
                        r = fut.result(timeout=0)
                    except Exception:
                        r = None
                    if r is not None:
                        raw.append(r)
                    else:
                        n_generation_fail += 1
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

    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    pbar.close()

    elapsed = time.perf_counter() - (t_deadline - _RUNTIME_S)

    samples = [
        Sample(nodes=r[0], edges=r[1], milp_ok=bool(r[2]),
               milp_flow=float(r[3]), sim_flow=float(r[4]),
               sim_converged=bool(r[5]))
        for r in raw
    ]

    n_total = len(samples)
    n_milp_ok = sum(1 for s in samples if s.milp_ok)
    n_converged = sum(1 for s in samples if s.milp_ok and s.sim_converged)

    print(f"\n  {elapsed:.0f}s  |  {n_submitted} submitted  |  {n_total} sampled")
    print(f"  MILP optimal: {n_milp_ok}/{n_total}  |  SIM converged: {n_converged}/{n_milp_ok}")
    print(f"  Generation failures: {n_generation_fail}\n")

    if n_milp_ok < 3:
        print("  Not enough MILP-ok samples to report.")
        return

    # ── bucket all MILP-ok samples ──────────────────────────────────────────

    ok_samples = [s for s in samples if s.milp_ok]
    fail_samples = [s for s in samples if not s.milp_ok]

    buckets: Dict[Tuple[int, int], List[Sample]] = {}
    fail_buckets: Dict[Tuple[int, int], List[Sample]] = {}

    for s in ok_samples:
        buckets.setdefault(_bkt_range(s.edges), []).append(s)
    for s in fail_samples:
        fail_buckets.setdefault(_bkt_range(s.edges), []).append(s)

    all_bkt_keys = sorted(set(buckets) | set(fail_buckets))

    stats_list: List[BucketStats] = []
    for (lo, hi) in all_bkt_keys:
        bucket = ok_samples if (lo, hi) in buckets else []
        if (lo, hi) in buckets:
            bucket = buckets[(lo, hi)]
        else:
            bucket = []
        fbucket = fail_buckets.get((lo, hi), [])
        total_ok = len(bucket)
        n_fail = len(fbucket)
        if total_ok + n_fail < _MIN_PER_BUCKET:
            continue

        n_conv = sum(1 for s in bucket if s.sim_converged)
        converge_rate = n_conv / total_ok if total_ok > 0 else 0.0

        errors = [abs(s.sim_flow - s.milp_flow) for s in bucket if s.sim_converged]
        mae = statistics.mean(errors) if errors else 0.0
        max_err = max(errors) if errors else 0.0
        n_agree = sum(1 for e in errors if e < _EPSILON)
        agree_rate = n_agree / len(errors) if errors else 0.0

        stats_list.append(BucketStats(
            edge_low=lo, edge_high=hi,
            n_total=total_ok,
            n_milp_fail=n_fail,
            n_converged=n_conv,
            converge_rate=converge_rate,
            mae=mae,
            max_error=max_err,
            agreement_rate=agree_rate,
        ))

    if not stats_list:
        print("  No buckets with enough samples.")
        return

    # ── global stats ────────────────────────────────────────────────────────

    all_errors = [abs(s.sim_flow - s.milp_flow) for s in ok_samples if s.sim_converged]
    global_mae = statistics.mean(all_errors) if all_errors else 0.0
    global_max = max(all_errors) if all_errors else 0.0
    global_agree = sum(1 for e in all_errors if e < _EPSILON)
    global_agree_rate = global_agree / len(all_errors) if all_errors else 0.0
    global_conv_rate = n_converged / n_milp_ok if n_milp_ok > 0 else 0.0

    # ── print table ─────────────────────────────────────────────────────────

    print(f"  Global  —  converged: {n_converged}/{n_milp_ok} ({global_conv_rate:.1%})")
    print(f"            MAE={global_mae:.6e}  |  max_err={global_max:.6e}")
    print(f"            agreement: {global_agree}/{len(all_errors)} ({global_agree_rate:.1%})  (error < {_EPSILON})")
    print()

    hdr = (f"  {'Edges':>8}  {'N':>5}  {'Fail':>5}  {'Conv%':>7}  "
           f"{'MAE':>10}  {'MaxErr':>10}  {'Agree%':>7}")
    print(hdr)
    print("  " + "-" * 65)
    for st in stats_list:
        lbl = _bkt_label(st.edge_low, st.edge_high)
        conv_pct = f"{st.converge_rate:.0%}" if st.converge_rate else "-"
        mae_str = f"{st.mae:.4e}" if st.mae else "-"
        mx_str = f"{st.max_error:.4e}" if st.max_error else "-"
        agree_str = f"{st.agreement_rate:.0%}" if st.agreement_rate else "-"
        print(f"  {lbl:>8}  {st.n_total:>5}  {st.n_milp_fail:>5}  {conv_pct:>7}  "
              f"{mae_str:>10}  {mx_str:>10}  {agree_str:>7}")

    # ── save JSON ───────────────────────────────────────────────────────────

    os.makedirs("output", exist_ok=True)

    samples_out = []
    for s in ok_samples:
        err = abs(s.sim_flow - s.milp_flow) if s.sim_converged else None
        samples_out.append({
            "nodes": s.nodes,
            "edges": s.edges,
            "milp_flow": round(s.milp_flow, 12),
            "sim_flow": round(s.sim_flow, 12) if s.sim_converged else None,
            "sim_converged": s.sim_converged,
            "error": round(err, 12) if err is not None else None,
        })

    json_out = {
        "runtime_s": round(elapsed),
        "n_submitted": n_submitted,
        "n_total": n_total,
        "n_milp_ok": n_milp_ok,
        "n_milp_fail": len(fail_samples),
        "n_gen_fail": n_generation_fail,
        "n_sim_converged": n_converged,
        "converge_rate": round(global_conv_rate, 4),
        "mae": global_mae,
        "max_error": global_max,
        "agreement_rate": round(global_agree_rate, 4),
        "agreement_threshold": _EPSILON,
        "buckets": [
            {
                "edge_low": s.edge_low, "edge_high": s.edge_high,
                "n_total": s.n_total, "n_milp_fail": s.n_milp_fail,
                "n_converged": s.n_converged,
                "converge_rate": round(s.converge_rate, 4),
                "mae": s.mae,
                "max_error": s.max_error,
                "agreement_rate": round(s.agreement_rate, 4),
            }
            for s in stats_list
        ],
        "samples": samples_out,
    }

    json_path = "output/validate_sim.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report saved → {json_path}")


if __name__ == "__main__":
    main()
