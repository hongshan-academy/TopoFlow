"""Benchmark simulation+fallback vs pure MILP to find optimal mixed_edge_threshold.

Compares two evaluation strategies per graph:
  1. simulation + MILP fallback (total wall time including fallback)
  2. pure MILP solver (always)

ProcessPoolExecutor (16 workers), runs ~12 minutes.
Fits exponential curves, models sim+fallback as piecewise exponential,
marks fail/timeout points outside the plot range.
"""

import gc
import json
import math
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
_RUNTIME_S = 720
_MAX_SIM_FRAMES = 100000
_EDGE_BUCKET = 5
_MIN_PER_BUCKET = 5
_SEED = 42
_MAX_EDGES = 85
_N_INTERNAL_RANGE = list(range(3, 55))

_CONV_THRESHOLD = 0.5   # convergence rate below which to split piecewise fit


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Sample:
    nodes: int
    edges: int
    milp_ok: bool              # did MILP solve optimally
    milp_ms: float             # pure MILP wall time (valid iff milp_ok)
    sim_fallback_ms: float     # sim total: converge? sim time : sim + MILP fallback
    sim_only_ms: float         # simulation-only wall time
    sim_converged: bool         # did simulation find a cycle


@dataclass
class BucketStats:
    edge_low: int
    edge_high: int
    n_total: int               # MILP-ok samples
    n_fail: int                # MILP-fail samples in this bucket
    n_converged: int           # how many sim converged
    milp_mean_ms: float
    milp_std_ms: float
    sim_fb_mean_ms: float
    sim_fb_std_ms: float
    sim_only_mean_ms: float    # sim-only mean (converged or not)
    converge_rate: float       # fraction that converged
    winner: str


# ── Worker ────────────────────────────────────────────────────────────────────

def _do_one(n_internal: int, seed_val: int) -> Optional[Tuple]:
    random.seed(seed_val)
    from ga.generation import generate_strict_graph
    from solver import solve
    from simulator import simulate
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

    # ── 1. Pure MILP ──
    milp_ok = False
    milp_ms = -1.0
    try:
        t0 = time.perf_counter()
        result = solve(graph, threads=1)
        milp_ms = (time.perf_counter() - t0) * 1000
        milp_ok = (result.status == _pulp.constants.LpStatusOptimal)
    except Exception:
        pass

    # ── 2. Simulation + fallback ──
    sim_converged = False
    sim_only_ms = -1.0
    sim_fallback_ms = -1.0
    try:
        t0_sim = time.perf_counter()
        sim_result = simulate(graph, max_frames=_MAX_SIM_FRAMES)
        sim_only_ms = (time.perf_counter() - t0_sim) * 1000
        sim_converged = sim_result.converged

        if sim_converged:
            sim_fallback_ms = sim_only_ms
        elif milp_ok:
            t0_fb = time.perf_counter()
            solve(graph, threads=1)
            fb_ms = (time.perf_counter() - t0_fb) * 1000
            sim_fallback_ms = sim_only_ms + fb_ms
        else:
            sim_fallback_ms = sim_only_ms
    except Exception:
        pass

    gc.collect()
    return (nodes, edges, int(milp_ok), milp_ms, sim_fallback_ms, sim_only_ms, int(sim_converged))


# ── Bucket helpers ────────────────────────────────────────────────────────────

def _bkt_range(e: int) -> Tuple[int, int]:
    lo = (e // _EDGE_BUCKET) * _EDGE_BUCKET
    return lo, lo + _EDGE_BUCKET - 1


def _bkt_mid(lo: int, hi: int) -> float:
    return (lo + hi + 1) / 2


def _bkt_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo:>3d}-{hi:>3d}"


# ── Exponential fitting ────────────────────────────────────────────────────────

def _exp_fit(xs, ys):
    """Fit y = a * exp(b * x). Returns (a, b, r2, xs_fine, ys_fine)."""
    import numpy as np

    arr_x = np.array(xs, dtype=np.float64)
    arr_y = np.array(ys, dtype=np.float64)
    log_y = np.log(np.maximum(arr_y, 1e-9))
    b, log_a = np.polyfit(arr_x, log_y, 1)
    a = math.exp(log_a)
    y_pred = a * np.exp(b * arr_x)
    ss_res = np.sum((arr_y - y_pred) ** 2)
    ss_tot = np.sum((arr_y - np.mean(arr_y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)

    xs_fine = np.linspace(xs[0], xs[-1], 200)
    ys_fine = a * np.exp(b * xs_fine)
    return a, b, r2, xs_fine, ys_fine


def _piecewise_exp_fit(buckets: List[BucketStats]):
    """Piecewise exponential fit for sim+fallback: split by convergence rate."""
    import numpy as np

    xs_all = np.array([_bkt_mid(s.edge_low, s.edge_high) for s in buckets])
    ys_all = np.array([s.sim_fb_mean_ms for s in buckets])
    crates = np.array([s.converge_rate for s in buckets])

    if len(buckets) < 3:
        return None

    # find split: last bucket where convergence >= CONV_THRESHOLD
    split_idx = 0
    for i in range(len(buckets)):
        if crates[i] >= _CONV_THRESHOLD:
            split_idx = i
        else:
            break

    # ensure at least 2 points per side
    if split_idx < 1:
        split_idx = 1
    if split_idx >= len(buckets) - 1:
        split_idx = len(buckets) - 2

    xs_lo = xs_all[:split_idx + 1]
    ys_lo = ys_all[:split_idx + 1]
    xs_hi = xs_all[split_idx:]
    ys_hi = ys_all[split_idx:]

    result = {"split_x": float(xs_all[split_idx]), "split_edges": buckets[split_idx].edge_low, "fits": []}

    if len(xs_lo) >= 2:
        a1, b1, r2_1, xf1, yf1 = _exp_fit(xs_lo.tolist(), ys_lo.tolist())
        result["fits"].append({"label": "convergent", "a": a1, "b": b1, "r2": r2_1,
                                "xs": xf1, "ys": yf1})

    if len(xs_hi) >= 2:
        a2, b2, r2_2, xf2, yf2 = _exp_fit(xs_hi.tolist(), ys_hi.tolist())
        result["fits"].append({"label": "divergent", "a": a2, "b": b2, "r2": r2_2,
                                "xs": xf2, "ys": yf2})

    return result


# ── Intersection ──────────────────────────────────────────────────────────────

def _find_intersection_from_fits(
    buckets: List[BucketStats],
    milp_fit,
    piecewise_fit,
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """Find crossover between fitted MILP curve and piecewise sim+fallback curve."""
    import numpy as np

    if milp_fit is None or piecewise_fit is None or not piecewise_fit["fits"]:
        return None, None

    milp_xs = np.array(milp_fit[3])
    milp_ys = np.array(milp_fit[4])

    for pw in piecewise_fit["fits"]:
        pw_xs = pw["xs"]
        pw_ys = pw["ys"]

        # linear interpolation in x-space
        diff = pw_ys - np.interp(pw_xs, milp_xs, milp_ys)
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            x_cross = float(pw_xs[idx])
            y_cross = float(pw_ys[idx])
            threshold = int(round(x_cross))
            return threshold, (x_cross, y_cross)

    # fallback: use bucket means
    xs = [_bkt_mid(s.edge_low, s.edge_high) for s in buckets]
    fb_means = [s.sim_fb_mean_ms for s in buckets]
    mp_means = [s.milp_mean_ms for s in buckets]

    for i in range(1, len(buckets)):
        prev_diff = fb_means[i - 1] - mp_means[i - 1]
        cur_diff = fb_means[i] - mp_means[i]
        if prev_diff <= 0 < cur_diff:
            x1, x2 = xs[i - 1], xs[i]
            t = abs(prev_diff) / (abs(prev_diff) + cur_diff)
            x_cross = x1 + t * (x2 - x1)
            y_cross = mp_means[i - 1] + (x_cross - x1) * (mp_means[i] - mp_means[i - 1]) / (x2 - x1)
            return int(round(x_cross)), (x_cross, y_cross)

    return None, None


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(
    buckets: List[BucketStats],
    milp_fit: Optional[Tuple],
    piecewise_fit: Optional[dict],
    cross_edges: Optional[float],
    cross_ms: Optional[float],
    threshold: Optional[int],
    elapsed: float,
    n_total: int,
    n_milp_ok: int,
    n_milp_fail: int,
    n_converged: int,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [_bkt_mid(s.edge_low, s.edge_high) for s in buckets]
    fb_means = [s.sim_fb_mean_ms for s in buckets]
    fb_stds = [s.sim_fb_std_ms for s in buckets]
    mp_means = [s.milp_mean_ms for s in buckets]
    mp_stds = [s.milp_std_ms for s in buckets]
    fail_counts = [s.n_fail for s in buckets]
    nonconv_fracs = [1.0 - s.converge_rate for s in buckets]

    fig, (ax_main, ax_fail) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05},
        sharex=True,
    )

    color_fb = "#d62728"
    color_mp = "#1f77b4"
    color_fail = "#7f7f7f"
    color_fb_fit = "#ff6b6b"
    color_mp_fit = "#4a9eff"

    # ── Main axes: raw data + fits ──

    # raw scatter with error bands
    ax_main.errorbar(xs, fb_means, yerr=fb_stds, fmt="o", color=color_fb,
                     capsize=3, markersize=5, alpha=0.55, label="sim+fallback (raw)")
    ax_main.errorbar(xs, mp_means, yerr=mp_stds, fmt="s", color=color_mp,
                     capsize=3, markersize=5, alpha=0.55, label="pure MILP (raw)")

    # fitted curves
    if milp_fit is not None:
        _, _, r2_m, xf_m, yf_m = milp_fit
        ax_main.plot(xf_m, yf_m, color=color_mp_fit, linewidth=2.5,
                     label=f"pure MILP fit  (R²={r2_m:.3f})")

    if piecewise_fit is not None and piecewise_fit["fits"]:
        for pw in piecewise_fit["fits"]:
            ls = "-" if pw["label"] == "convergent" else "--"
            lbl = f"sim+FB {pw['label']}  (R²={pw['r2']:.3f})"
            ax_main.plot(pw["xs"], pw["ys"], color=color_fb_fit, linewidth=2.5,
                         linestyle=ls, label=lbl)

    # crossover
    if cross_edges is not None and cross_ms is not None:
        ax_main.axvline(x=cross_edges, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_main.annotate(
            f"  crossover ≈ {cross_edges:.1f} edges\n  threshold = {threshold}",
            xy=(cross_edges, cross_ms),
            xytext=(cross_edges + 4, cross_ms * 1.4),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    ax_main.set_ylabel("Time (ms)", fontsize=11)
    ax_main.legend(fontsize=8, loc="upper left", ncol=2)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_yscale("log")

    # y-max annotation for fail markers
    ylim = ax_main.get_ylim()
    y_max = ylim[1]

    if threshold is not None and xs:
        ax_main.axvspan(0, threshold, color="green", alpha=0.03, label="_nolegend_")
        mid_y = y_max / 2
        y_for_text = y_max * 0.5
        ax_main.text(threshold / 2, y_for_text,
                     "simulation", fontsize=10, color="green", ha="center", va="center",
                     fontstyle="italic", alpha=0.8)
        ax_main.text((threshold + xs[-1]) / 2, y_for_text,
                     "MILP", fontsize=10, color="blue", ha="center", va="center",
                     fontstyle="italic", alpha=0.8)

    # ── Fail/timed-out panel ──

    bar_w = (_EDGE_BUCKET - 1) * 0.8
    x_bar = xs

    # stack: fail count, nonconverged fraction as second bar
    total_per_bkt = [s.n_total + s.n_fail for s in buckets]
    fail_fracs = [s.n_fail / max(t, 1) for s in buckets]

    ax_fail.bar(x_bar, fail_fracs, width=bar_w, color="lightcoral", edgecolor=color_fail,
                linewidth=0.8, alpha=0.8, label="MILP fail fraction")

    # second bar showing non-converged fraction (of MILP-ok samples)
    nonconv_f = [min(nc, 1.0) for nc in nonconv_fracs]
    ax_fail.bar(x_bar, nonconv_f, width=bar_w * 0.6, color="orange", edgecolor="darkorange",
                linewidth=0.8, alpha=0.7, label="SIM timed-out fraction")

    # annotate fail counts
    for i, (x, fc, nc) in enumerate(zip(x_bar, fail_counts, nonconv_fracs)):
        if fc > 0:
            ax_fail.text(x, fail_fracs[i] + 0.02, str(fc), ha="center", fontsize=7, color="darkred")
        nc_count = buckets[i].n_total - buckets[i].n_converged
        if nc_count > 0:
            ax_fail.text(x, nonconv_f[i] + 0.02, str(nc_count), ha="center", fontsize=7, color="darkorange")

    ax_fail.set_ylabel("Fraction", fontsize=9)
    ax_fail.set_xlabel("Number of edges", fontsize=11)
    ax_fail.set_ylim(-0.05, 1.15)
    ax_fail.legend(fontsize=8, loc="upper right", ncol=2)
    ax_fail.grid(True, alpha=0.3, axis="y")
    ax_fail.set_xlim(0, xs[-1] + _EDGE_BUCKET / 2 if xs else _MAX_EDGES)

    fig.suptitle(
        f"MILP vs Simulation+Fallback by Edge Count  |  "
        f"{n_total} samples ({n_milp_ok} MILP-ok, {n_milp_fail} fail, {n_converged} sim converged)  |  "
        f"{elapsed:.0f}s",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = "output/benchmark_threshold.png"
    os.makedirs("output", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("  TopoFlow Solver Benchmark — Threshold Search (exp fit)")
    print(f"  Workers: {_WORKERS}  |  Runtime: ~{_RUNTIME_S}s  |  Sim max frames: {_MAX_SIM_FRAMES}")
    print(f"  Bucket: {_EDGE_BUCKET} edges  |  Max edges: {_MAX_EDGES}  |  Min/bucket: {_MIN_PER_BUCKET}")
    print(f"  Piecewise split where converge rate < {_CONV_THRESHOLD:.0%}")
    print("=" * 80)

    rng = random.Random(_SEED)
    seed_base = _SEED * 1000000
    task_iter = 0

    raw: List[Tuple] = []
    fail_raw: List[Tuple] = []
    n_submitted = 0
    n_generation_fail = 0
    t_deadline = time.perf_counter() + _RUNTIME_S

    pbar = tqdm.tqdm(total=None, desc="  Benchmark", unit="t", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        futures: Dict = {}

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
            n_milp_fail = len(fail_raw)
            pbar.set_postfix_str(
                f"ok={n_milp_ok} fail={n_milp_fail} sub={n_submitted} "
                f"pend={len(futures)} {remaining:.0f}s"
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
                        if r[2]:  # milp_ok
                            raw.append(r)
                        else:
                            fail_raw.append(r)
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

        for fut in list(futures):
            fut.cancel()

    pbar.close()

    elapsed = time.perf_counter() - (t_deadline - _RUNTIME_S)
    samples = [
        Sample(nodes=r[0], edges=r[1], milp_ok=bool(r[2]), milp_ms=r[3],
               sim_fallback_ms=r[4], sim_only_ms=r[5],
               sim_converged=bool(r[6]))
        for r in raw
    ]
    fail_samples = [
        Sample(nodes=r[0], edges=r[1], milp_ok=False, milp_ms=-1.0,
               sim_fallback_ms=r[4], sim_only_ms=r[5],
               sim_converged=bool(r[6] if len(r) > 6 else False))
        for r in fail_raw
    ]

    n_converged = sum(1 for s in samples if s.sim_converged)
    n_milp_ok = len(samples)
    n_milp_fail = len(fail_samples)
    print(f"\n  {elapsed:.0f}s  |  {n_milp_ok} MILP-ok  |  {n_milp_fail} MILP-fail"
          f"  |  gen_fail={n_generation_fail}"
          f"  |  SIM converged={n_converged}/{n_milp_ok}\n")

    if not samples:
        print("  No valid MILP-ok samples.")
        return

    # ── bucket all samples ─────────────────────────────────────────────────────

    buckets: Dict[Tuple[int, int], List[Sample]] = {}
    fail_buckets: Dict[Tuple[int, int], List[Sample]] = {}

    for s in samples:
        buckets.setdefault(_bkt_range(s.edges), []).append(s)
    for s in fail_samples:
        fail_buckets.setdefault(_bkt_range(s.edges), []).append(s)

    all_bkt_keys = sorted(set(buckets) | set(fail_buckets))

    stats_list: List[BucketStats] = []
    for (lo, hi) in all_bkt_keys:
        bucket = buckets.get((lo, hi), [])
        fbucket = fail_buckets.get((lo, hi), [])
        total_ok = len(bucket)
        n_fail = len(fbucket)
        if total_ok + n_fail < _MIN_PER_BUCKET:
            continue

        n_conv = sum(1 for s in bucket if s.sim_converged)

        milp_pts = [s.milp_ms for s in bucket]
        fb_pts = [s.sim_fallback_ms for s in bucket]
        sim_only_pts = [s.sim_only_ms for s in bucket]

        milp_mean = statistics.mean(milp_pts) if milp_pts else 0.0
        milp_std = statistics.stdev(milp_pts) if len(milp_pts) >= 2 else 0.0
        fb_mean = statistics.mean(fb_pts) if fb_pts else 0.0
        fb_std = statistics.stdev(fb_pts) if len(fb_pts) >= 2 else 0.0
        sim_only_mean = statistics.mean(sim_only_pts) if sim_only_pts else 0.0
        converge_rate = n_conv / total_ok if total_ok > 0 else 0.0

        winner = "sim" if fb_mean <= milp_mean else "milp"

        stats_list.append(BucketStats(
            edge_low=lo, edge_high=hi,
            n_total=total_ok,
            n_fail=n_fail,
            n_converged=n_conv,
            milp_mean_ms=milp_mean, milp_std_ms=milp_std,
            sim_fb_mean_ms=fb_mean, sim_fb_std_ms=fb_std,
            sim_only_mean_ms=sim_only_mean,
            converge_rate=converge_rate,
            winner=winner,
        ))

    if not stats_list:
        print("  No buckets with enough samples.")
        return

    # ── exponential fits ──────────────────────────────────────────────────────

    xs_fit = [float(_bkt_mid(s.edge_low, s.edge_high)) for s in stats_list]

    milp_fit = _exp_fit(xs_fit, [s.milp_mean_ms for s in stats_list])
    _, _, r2_milp, _, _ = milp_fit

    piecewise_fit = _piecewise_exp_fit(stats_list)

    threshold, (cross_x, cross_y) = _find_intersection_from_fits(
        stats_list, milp_fit, piecewise_fit,
    )

    # ── print table ───────────────────────────────────────────────────────────

    print(f"  {'Edges':>10}  {'N':>4}  {'Fail':>5}  {'Conv%':>6}  "
          f"{'MILP mean':>10}  {'Sim+FB mean':>12}  {'Winner':>8}")
    print("  " + "-" * 68)
    for st in stats_list:
        lbl = _bkt_label(st.edge_low, st.edge_high)
        conv_pct = f"{st.converge_rate:.0%}"
        w = "MILP" if st.winner == "milp" else "SIM "
        print(f"  {lbl:>10}  {st.n_total:>4}  {st.n_fail:>5}  {conv_pct:>6}  "
              f"{st.milp_mean_ms:>8.1f}ms  {st.sim_fb_mean_ms:>10.1f}ms  "
              f"{w:>8}")

    # ── fit summary ───────────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"  MILP fit:    y = {milp_fit[0]:.4f} * exp({milp_fit[1]:.6f} * x)   R² = {r2_milp:.4f}")
    if piecewise_fit and piecewise_fit["fits"]:
        for pw in piecewise_fit["fits"]:
            print(f"  Sim+FB {pw['label']:>10}:  y = {pw['a']:.4f} * exp({pw['b']:.6f} * x)"
                  f"   R² = {pw['r2']:.4f}")
        print(f"  Piecewise split at ~{piecewise_fit['split_edges']} edges"
              f" (converge rate < {_CONV_THRESHOLD:.0%})")
    print(f"{'=' * 80}")

    # ── intersection ──────────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    if threshold is not None:
        first_winner = stats_list[0].winner
        print(f"  Crossover at {cross_x:.1f} edges  →  recommended threshold = {threshold}")
        if first_winner == "sim":
            print(f"  edges <= {threshold} → simulation+fallback  |  edges > {threshold} → pure MILP")
        else:
            print(f"  edges <= {threshold} → pure MILP  |  edges > {threshold} → simulation+fallback")
    else:
        wn = stats_list[0].winner
        print(f"  No crossover found — {wn} is faster across all buckets")
        threshold = 0 if wn == "sim" else 999
    print(f"  mixed_edge_threshold = {threshold}")
    print(f"{'=' * 80}")

    # ── save plot ─────────────────────────────────────────────────────────────

    plot_path = _plot(
        stats_list, milp_fit, piecewise_fit,
        cross_x, cross_y, threshold, elapsed,
        len(samples) + len(fail_samples), n_milp_ok, n_milp_fail, n_converged,
    )
    print(f"\n  Plot saved → {plot_path}")

    # ── save JSON ─────────────────────────────────────────────────────────────

    os.makedirs("output", exist_ok=True)
    json_out = {
        "runtime_s": round(elapsed),
        "n_submitted": n_submitted,
        "n_total": n_milp_ok + n_milp_fail,
        "n_milp_ok": n_milp_ok,
        "n_milp_fail": n_milp_fail,
        "n_gen_fail": n_generation_fail,
        "n_sim_converged": n_converged,
        "threshold": threshold,
        "crossover_edges": round(cross_x, 2) if cross_x is not None else None,
        "milp_fit": {"a": round(milp_fit[0], 6), "b": round(milp_fit[1], 6), "r2": round(r2_milp, 4)},
        "piecewise_fit": None,
        "buckets": [
            {"edge_low": s.edge_low, "edge_high": s.edge_high,
             "n_total": s.n_total, "n_fail": s.n_fail,
             "n_converged": s.n_converged,
             "converge_rate": round(s.converge_rate, 3),
             "milp_mean_ms": round(s.milp_mean_ms, 2),
             "milp_std_ms": round(s.milp_std_ms, 2),
             "sim_fb_mean_ms": round(s.sim_fb_mean_ms, 2),
             "sim_fb_std_ms": round(s.sim_fb_std_ms, 2),
             "sim_only_mean_ms": round(s.sim_only_mean_ms, 2),
             "winner": s.winner}
            for s in stats_list
        ],
    }

    if piecewise_fit and piecewise_fit["fits"]:
        pw_out = {
            "split_x": round(piecewise_fit["split_x"], 1),
            "split_edges": piecewise_fit["split_edges"],
            "fits": [
                {"label": f["label"],
                 "a": round(f["a"], 6), "b": round(f["b"], 6), "r2": round(f["r2"], 4)}
                for f in piecewise_fit["fits"]
            ],
        }
        json_out["piecewise_fit"] = pw_out

    with open("output/benchmark_threshold.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"  Data saved → output/benchmark_threshold.json")


if __name__ == "__main__":
    main()
