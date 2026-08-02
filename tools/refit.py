"""Post-hoc exponential fitting + replot from existing benchmark data.

Reads output/benchmark_threshold.json and produces:
  - Exponential fit for pure MILP
  - Piecewise exponential fit for sim+fallback (split by converge rate)
  - Plot with fitted curves, raw data, fail/timeout markers
  - Updated JSON with fit parameters
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EDGE_BUCKET = 5
_CONV_THRESHOLD = 0.5


def _bkt_mid(lo: int, hi: int) -> float:
    return (lo + hi + 1) / 2


def _exp_fit(xs, ys):
    """Fit y = a * exp(b * x). Returns (a, b, r2, xs_fine, ys_fine)."""
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


def _piecewise_exp_fit(buckets: List[dict]):
    """Piecewise exponential fit: split where converge_rate < threshold."""
    xs_all = np.array([_bkt_mid(b["edge_low"], b["edge_high"]) for b in buckets])
    ys_all = np.array([b["sim_fb_mean_ms"] for b in buckets])
    crates = np.array([b["n_converged"] / max(b["n_total"], 1) for b in buckets])

    if len(buckets) < 3:
        return None

    split_idx = 0
    for i in range(len(buckets)):
        if crates[i] >= _CONV_THRESHOLD:
            split_idx = i
        else:
            break

    if split_idx < 1:
        split_idx = 1
    if split_idx >= len(buckets) - 1:
        split_idx = len(buckets) - 2

    xs_lo = xs_all[:split_idx + 1]
    ys_lo = ys_all[:split_idx + 1]
    xs_hi = xs_all[split_idx:]
    ys_hi = ys_all[split_idx:]

    result = {
        "split_x": float(xs_all[split_idx]),
        "split_edges": buckets[split_idx]["edge_low"],
        "fits": [],
    }

    if len(xs_lo) >= 2:
        a1, b1, r2_1, xf1, yf1 = _exp_fit(xs_lo.tolist(), ys_lo.tolist())
        result["fits"].append({
            "label": "convergent", "a": a1, "b": b1, "r2": r2_1,
            "xs": xf1.tolist(), "ys": yf1.tolist(),
        })

    if len(xs_hi) >= 2:
        a2, b2, r2_2, xf2, yf2 = _exp_fit(xs_hi.tolist(), ys_hi.tolist())
        result["fits"].append({
            "label": "divergent", "a": a2, "b": b2, "r2": r2_2,
            "xs": xf2.tolist(), "ys": yf2.tolist(),
        })

    return result


def _find_crossover(
    buckets: List[dict],
    milp_fit: dict,
    piecewise_fit: dict,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Find crossover between fitted MILP and piecewise sim+fallback curves."""
    if piecewise_fit is None or not piecewise_fit.get("fits"):
        return None, None, None

    milp_xs = np.array(milp_fit["xs"])
    milp_ys = np.array(milp_fit["ys"])

    for pw in piecewise_fit["fits"]:
        pw_xs = np.array(pw["xs"])
        pw_ys = np.array(pw["ys"])

        diff = pw_ys - np.interp(pw_xs, milp_xs, milp_ys)
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            x_cross = float(pw_xs[idx])
            y_cross = float(pw_ys[idx])
            threshold = int(round(x_cross))
            return threshold, x_cross, y_cross

    # fallback: bucket means
    xs = [_bkt_mid(b["edge_low"], b["edge_high"]) for b in buckets]
    fb = [b["sim_fb_mean_ms"] for b in buckets]
    mp = [b["milp_mean_ms"] for b in buckets]

    for i in range(1, len(buckets)):
        pd = fb[i - 1] - mp[i - 1]
        cd = fb[i] - mp[i]
        if pd <= 0 < cd:
            x1, x2 = xs[i - 1], xs[i]
            t = abs(pd) / (abs(pd) + cd)
            x_cross = x1 + t * (x2 - x1)
            y_cross = mp[i - 1] + (x_cross - x1) * (mp[i] - mp[i - 1]) / (x2 - x1)
            return int(round(x_cross)), x_cross, y_cross

    return None, None, None


def plot(
    buckets: List[dict],
    milp_fit: dict,
    piecewise_fit: dict,
    threshold: Optional[int],
    cross_x: Optional[float],
    cross_y: Optional[float],
    n_total: int,
    n_converged: int,
    n_milp_fail: int,
    elapsed: int,
) -> str:
    xs = [_bkt_mid(b["edge_low"], b["edge_high"]) for b in buckets]
    fb_means = [b["sim_fb_mean_ms"] for b in buckets]
    fb_stds = [b["sim_fb_std_ms"] for b in buckets]
    mp_means = [b["milp_mean_ms"] for b in buckets]
    mp_stds = [b["milp_std_ms"] for b in buckets]
    crates = [b["n_converged"] / max(b["n_total"], 1) for b in buckets]
    nonconv_counts = [b["n_total"] - b["n_converged"] for b in buckets]

    fig, (ax_main, ax_fail) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05},
        sharex=True,
    )

    color_fb = "#d62728"
    color_mp = "#1f77b4"
    color_fb_fit = "#ff4444"
    color_mp_fit = "#3388cc"

    # ── raw data ──
    ax_main.errorbar(xs, fb_means, yerr=fb_stds, fmt="o", color=color_fb,
                     capsize=3, markersize=5, alpha=0.5,
                     label="sim+fallback (raw)")
    ax_main.errorbar(xs, mp_means, yerr=mp_stds, fmt="s", color=color_mp,
                     capsize=3, markersize=5, alpha=0.5,
                     label="pure MILP (raw)")

    # ── fitted curves ──
    ax_main.plot(milp_fit["xs"], milp_fit["ys"], color=color_mp_fit, linewidth=2.5,
                 label=f"pure MILP fit  $y = {milp_fit['a']:.3f} \\cdot e^{{{milp_fit['b']:.5f}x}}$  "
                       f"($R^2$={milp_fit['r2']:.3f})")

    if piecewise_fit and piecewise_fit.get("fits"):
        for pw in piecewise_fit["fits"]:
            ls = "-" if pw["label"] == "convergent" else "--"
            lbl = (f"sim+FB {pw['label']}  "
                   f"$y = {pw['a']:.3f} \\cdot e^{{{pw['b']:.5f}x}}$  "
                   f"($R^2$={pw['r2']:.3f})")
            ax_main.plot(pw["xs"], pw["ys"], color=color_fb_fit, linewidth=2.5,
                         linestyle=ls, label=lbl)

    # ── crossover ──
    if cross_x is not None and cross_y is not None:
        ax_main.axvline(x=cross_x, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_main.annotate(
            f"  crossover = {cross_x:.1f} edges\n  threshold = {threshold}",
            xy=(cross_x, cross_y),
            xytext=(cross_x + 4, cross_y * 1.5),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    ax_main.set_ylabel("Time (ms)", fontsize=11)
    ax_main.set_yscale("log")
    ax_main.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_main.grid(True, alpha=0.3)

    ylim = ax_main.get_ylim()
    y_max = ylim[1]

    if threshold is not None and xs:
        ax_main.axvspan(0, threshold, color="green", alpha=0.03, label="_nolegend_")
        y_for_text = y_max * 0.3
        ax_main.text(threshold / 2, y_for_text,
                     "simulation", fontsize=11, color="green", ha="center", va="center",
                     fontstyle="italic", alpha=0.7)
        ax_main.text((threshold + xs[-1]) / 2, y_for_text,
                     "MILP", fontsize=11, color="blue", ha="center", va="center",
                     fontstyle="italic", alpha=0.7)

    # ── Fail/timed-out panel ──
    bar_w = (_EDGE_BUCKET - 1) * 0.8

    # non-converged fraction (timeout)
    ax_fail.bar(xs, nonconv_counts, width=bar_w, color="orange", edgecolor="darkorange",
                linewidth=0.8, alpha=0.7, label="SIM timed-out count")

    # annotate convergence counts
    for i, (x, nc, n_total) in enumerate(zip(xs, nonconv_counts, [b["n_total"] for b in buckets])):
        if nc > 0:
            rate = nc / n_total
            ax_fail.text(x, nc + max(nonconv_counts) * 0.03, f"{nc}/{n_total}", ha="center",
                         fontsize=6.5, color="darkorange", fontweight="bold")

    # MILP-fail annotation (global, since not per-bucket in old data)
    if n_milp_fail > 0:
        ax_fail.annotate(
            f"  MILP fail: {n_milp_fail} total (excluded)",
            xy=(0.98, 0.92), xycoords="axes fraction",
            fontsize=8, color="darkred", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="darkred", alpha=0.85),
        )

    ax_fail.set_ylabel("Count", fontsize=9)
    ax_fail.set_xlabel("Number of edges", fontsize=11)
    ax_fail.legend(fontsize=8, loc="upper right")
    ax_fail.grid(True, alpha=0.3, axis="y")
    ax_fail.set_xlim(0, xs[-1] + _EDGE_BUCKET / 2)
    ax_fail.set_ylim(bottom=-0.5)

    fig.suptitle(
        f"MILP vs Simulation+Fallback — Exponential Fits  |  "
        f"{n_total} samples, {n_converged} sim converged, {n_milp_fail} MILP fail  |  "
        f"{elapsed}s",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = "output/benchmark_threshold.png"
    os.makedirs("output", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/benchmark_threshold.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets: List[dict] = data["buckets"]
    n_total = data.get("n_total", sum(b["n_total"] for b in buckets))
    n_converged = data.get("n_sim_converged", sum(b["n_converged"] for b in buckets))
    n_milp_fail = data.get("n_milp_fail", 0)
    elapsed = data.get("runtime_s", 0)

    xs_fit = [_bkt_mid(b["edge_low"], b["edge_high"]) for b in buckets]

    # ── MILP exponential fit ──
    a_m, b_m, r2_m, xf_m, yf_m = _exp_fit(
        xs_fit, [b["milp_mean_ms"] for b in buckets],
    )
    milp_fit_out = {
        "a": a_m, "b": b_m, "r2": r2_m,
        "xs": xf_m.tolist(), "ys": yf_m.tolist(),
    }

    # ── Piecewise sim+fallback fit ──
    piecewise_fit = _piecewise_exp_fit(buckets)

    # ── Crossover ──
    threshold, cross_x, cross_y = _find_crossover(buckets, milp_fit_out, piecewise_fit)

    # ── Print ──
    print(f"MILP fit:       y = {a_m:.4f} * exp({b_m:.6f} * x)   R2 = {r2_m:.4f}")
    if piecewise_fit and piecewise_fit.get("fits"):
        for pw in piecewise_fit["fits"]:
            print(f"Sim+FB {pw['label']:>10}: y = {pw['a']:.4f} * exp({pw['b']:.6f} * x)"
                  f"   R2 = {pw['r2']:.4f}")
        print(f"Piecewise split at ~{piecewise_fit['split_edges']} edges")
    if threshold is not None:
        print(f"Crossover: {cross_x:.1f} edges  →  threshold = {threshold}")
    else:
        print("No crossover found")

    # ── Plot ──
    plot_path = plot(
        buckets, milp_fit_out, piecewise_fit,
        threshold, cross_x, cross_y,
        n_total, n_converged, n_milp_fail, elapsed,
    )
    print(f"Plot saved → {plot_path}")

    # ── Update JSON ──
    data["milp_fit"] = {"a": round(a_m, 6), "b": round(b_m, 6), "r2": round(r2_m, 4)}
    if piecewise_fit:
        pw_out = {
            "split_x": round(piecewise_fit["split_x"], 1),
            "split_edges": piecewise_fit["split_edges"],
            "fits": [
                {"label": f["label"], "a": round(f["a"], 6), "b": round(f["b"], 6),
                 "r2": round(f["r2"], 4)}
                for f in piecewise_fit["fits"]
            ],
        }
        data["piecewise_fit"] = pw_out
    new_threshold = threshold if threshold is not None else data.get("threshold")
    data["threshold"] = new_threshold
    data["crossover_edges"] = round(cross_x, 2) if cross_x is not None else None

    out_path = input_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated JSON → {out_path}")


if __name__ == "__main__":
    main()
