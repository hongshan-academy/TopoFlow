"""
Plots GA training history results.

Usage:
    python tools/plot.py                              # default output/ga_history.json
    python tools/plot.py --history output/ga_history.json
    python tools/plot.py --output training_report.png
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

_COLORS = {
    "best": "#1f77b4",
    "avg": "#ff7f0e",
    "nodes": "#2ca02c",
    "edges": "#d62728",
    "mutated": "#9467bd",
    "selected": "#17becf",
    "immigrants": "#bcbd22",
    "pending": "#e377c2",
    "ready": "#8c564b",
    "error": "#d62728",
}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _figsize(ncols: int = 2, nrows: int = 3) -> Tuple[float, float]:
    return (7 * ncols, 4.5 * nrows)


def plot_history(
    history: Dict[str, Any],
    output: str = "output/training_report.png",
) -> str:
    generations = history.get("generations", [])
    best_chain = history.get("best_chain", [])

    if not generations:
        print("No generation data found in history.")
        return output

    target = history.get("target", {})
    params = history.get("params", {})
    elapsed = history.get("elapsed_sec", 0)

    gens = [g["gen"] for g in generations]
    best_errs = [g["best_error"] for g in generations]
    avg_errs = [g["avg_error"] for g in generations]
    best_nodes_list = [g["best_nodes"] for g in generations]
    min_nodes_list = [g["min_nodes"] for g in generations]
    max_nodes_list = [g["max_nodes"] for g in generations]
    max_edges_list = [g["max_edges"] for g in generations]
    n_mutated_list = [g["n_mutated"] for g in generations]
    n_selected_list = [g["n_selected"] for g in generations]
    n_immigrants_list = [g["n_immigrants"] for g in generations]
    n_ready_list = [g["n_ready"] for g in generations]
    n_pending_list = [g["n_pending"] for g in generations]
    elapsed_list = [g["elapsed_sec"] for g in generations]
    cum_elapsed = []
    s = 0.0
    for e in elapsed_list:
        s += e
        cum_elapsed.append(s)

    nrows = 3
    fig, axes = plt.subplots(nrows, 2, figsize=_figsize(nrows=nrows))
    axes = axes.flatten()

    suptitle_parts = []
    if target:
        suptitle_parts.append(
            f"GA Training Report  |  Target = {target.get('p','?')}/{target.get('q','?')}"
        )
    suptitle_parts.append(f"  {len(generations)} generations")
    suptitle_parts.append(f"  {elapsed:.1f}s total")
    if params.get("pop_size"):
        suptitle_parts.append(f"  pop={params['pop_size']}")
    fig.suptitle(" ".join(suptitle_parts), fontsize=13, fontweight="bold")

    # ── 1. Error over generations ──
    ax = axes[0]
    color_best = _COLORS["best"]
    color_avg = _COLORS["avg"]
    ax.plot(gens, best_errs, color=color_best, linewidth=1.2, label="best error")
    ax.plot(gens, avg_errs, color=color_avg, linewidth=0.8, alpha=0.7, label="avg error")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Error  (log scale)")
    ax.set_title("Fitness Error Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")

    # ── 2. Best nodes over generations ──
    ax = axes[1]
    ax.plot(gens, best_nodes_list, color=_COLORS["nodes"], linewidth=1.2, label="best nodes")
    ax.fill_between(
        gens, min_nodes_list, max_nodes_list,
        color=_COLORS["nodes"], alpha=0.15, label="min–max range"
    )
    ax.plot(gens, max_edges_list, color=_COLORS["edges"], linewidth=1.0, alpha=0.7, label="max edges")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title("Best Nodes Over Generations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. Population composition ──
    ax = axes[2]
    ax.stackplot(
        gens,
        n_mutated_list,
        n_selected_list,
        n_immigrants_list,
        n_pending_list,
        labels=["mutated", "selected", "immigrants", "pending"],
        colors=[_COLORS["mutated"], _COLORS["selected"], _COLORS["immigrants"], _COLORS["pending"]],
        alpha=0.75,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Individuals")
    ax.set_title("Population Composition per Generation")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. Elapsed time ──
    ax = axes[3]
    ax.bar(gens, elapsed_list, color=_COLORS["best"], width=0.8, alpha=0.8)
    ax_twin = ax.twinx()
    ax_twin.plot(gens, cum_elapsed, color=_COLORS["error"], linewidth=1.5, label="cumulative")
    ax_twin.set_ylabel("Cumulative time (s)", color=_COLORS["error"])
    ax_twin.tick_params(axis="y", labelcolor=_COLORS["error"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Generation time (s)")
    ax.set_title("Elapsed Time per Generation")
    ax_twin.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 5. Best chain fitness ──
    ax_idx = 4
    if best_chain:
        bc_gens = [s["gen"] for s in best_chain]
        bc_errors = [s["fitness"][0] for s in best_chain]
        ax = axes[ax_idx]
        ax.step(
            bc_gens, bc_errors, where="post",
            color=_COLORS["error"], linewidth=1.5, label="best error"
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Error")
        ax.set_title(f"Best Chain Progression  ({len(best_chain)} improvements)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

        bc_nodes = [s["graph_nodes"] for s in best_chain]
        bc_edges = [s["graph_edges"] for s in best_chain]
        ax = axes[ax_idx] if ax_idx < len(axes) else None
        if ax is not None:
            ax.step(
                bc_gens, bc_nodes, where="post",
                color=_COLORS["nodes"], linewidth=1.2, label="nodes"
            )
            ax.step(
                bc_gens, bc_edges, where="post",
                color=_COLORS["edges"], linewidth=1.2, label="edges"
            )
            ax.set_xlabel("Generation")
            ax.set_ylabel("Count")
            ax.set_title("Best Chain Graph Size")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GA training results")
    parser.add_argument(
        "--history", type=str, default="output/ga_history.json",
        help="Path to GA history JSON (default: output/ga_history.json)",
    )
    parser.add_argument(
        "--output", type=str, default="output/training_report.png",
        help="Output image path (default: output/training_report.png)",
    )
    args = parser.parse_args()

    history = _load_json(args.history)

    plot_history(history, output=args.output)


if __name__ == "__main__":
    main()
