"""
GNN surrogate model visualization.

Usage:
    python tools/nn_viz.py
    python tools/nn_viz.py --model output/gnn_model.pt --samples output/gnn_samples.pkl
    python tools/nn_viz.py --output output/gnn_report.png
"""

import argparse
import os
import pickle
import sys
from typing import Any, List, Tuple

import numpy as np

_tools_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tools_dir)
if _tools_dir in sys.path:
    sys.path.remove(_tools_dir)
sys.path.insert(0, _project_root)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from nn_surrogate.model import SurrogateGNN
from nn_surrogate.data import build_predict_list


def _load_gnn(model_path: str) -> SurrogateGNN:
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    gnn = SurrogateGNN()
    gnn.load(model_path)
    if not gnn.is_ready():
        print("GNN model loaded but not trained")
        sys.exit(1)
    return gnn


def _load_samples(path: str) -> List[Tuple]:
    if not os.path.exists(path):
        print(f"Samples file not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_gnn_report(
    model: SurrogateGNN,
    samples: List[Tuple],
    output: str = "output/gnn_report.png",
) -> str:
    if len(samples) == 0:
        print("No archive data found.")
        return output

    edges_list = [s[0] for s in samples]
    y = np.array([s[1] for s in samples], dtype=np.float64)

    data_list = build_predict_list(edges_list)
    predictions = model.predict(data_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(y, predictions, alpha=0.35, s=8, edgecolors="none")
    lim_min = min(y.min(), predictions.min())
    lim_max = max(y.max(), predictions.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
            "r--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Actual Error")
    ax.set_ylabel("Predicted Error")
    ax.set_title(f"GNN Predictions vs Actual  (n_samples={len(samples)})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = predictions - y
    ax.hist(residuals, bins=50, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="r", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Residuals Distribution  "
        f"(mu={residuals.mean():.4f}, sigma={residuals.std():.4f})"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "GNN Surrogate Report",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"GNN report saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GNN surrogate model report")
    parser.add_argument(
        "--model", type=str, default="output/gnn_model.pt",
        help="Path to saved GNN model (default: output/gnn_model.pt)",
    )
    parser.add_argument(
        "--samples", type=str, default="output/gnn_samples.pkl",
        help="Path to raw samples pickle (default: output/gnn_samples.pkl)",
    )
    parser.add_argument(
        "--output", type=str, default="output/gnn_report.png",
        help="Output image path (default: output/gnn_report.png)",
    )
    args = parser.parse_args()

    model = _load_gnn(args.model)
    samples = _load_samples(args.samples)
    plot_gnn_report(model, samples, output=args.output)


if __name__ == "__main__":
    main()
