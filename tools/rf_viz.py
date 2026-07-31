"""
RF surrogate model visualization.

Usage:
    python tools/rf_viz.py
    python tools/rf_viz.py --model output/rf_model.pkl --X output/rf_archive_X.npy --y output/rf_archive_y.npy
    python tools/rf_viz.py --output output/rf_report.png
"""

import argparse
import os
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from rf_surrogate.features import FEATURE_NAMES


def _load_rf(model_path: str) -> Any:
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        print(f"Invalid data in {path}")
        sys.exit(1)
    return arr


def plot_rf_report(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    output: str = "output/rf_report.png",
) -> str:
    if X.shape[0] == 0:
        print("No archive data found.")
        return output

    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    sorted_imps = importances[indices]

    predictions = model.predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    ax = axes[0]
    colors = plt.cm.viridis(sorted_imps / max(sorted_imps, default=1))
    ax.barh(range(len(sorted_imps)), sorted_imps, color=colors)
    ax.set_yticks(range(len(sorted_imps)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"RF Feature Importance  (n_samples={X.shape[0]})")
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    ax.scatter(y, predictions, alpha=0.35, s=8, edgecolors="none")
    lim_min = min(y.min(), predictions.min())
    lim_max = max(y.max(), predictions.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
            "r--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Actual Error")
    ax.set_ylabel("Predicted Error")
    ax.set_title("RF Predictions vs Actual")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    residuals = predictions - y
    ax.hist(residuals, bins=50, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="r", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Residuals Distribution  "
        f"(μ={residuals.mean():.2f}, σ={residuals.std():.2f})"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"RF Surrogate Report  |  n_estimators={model.n_estimators}  "
        f"max_depth={model.max_depth}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"RF report saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RF surrogate model report")
    parser.add_argument(
        "--model", type=str, default="output/rf_model.pkl",
        help="Path to pickled RF model (default: output/rf_model.pkl)",
    )
    parser.add_argument(
        "--X", type=str, default="output/rf_archive_X.npy",
        help="Path to archive features .npy (default: output/rf_archive_X.npy)",
    )
    parser.add_argument(
        "--y", type=str, default="output/rf_archive_y.npy",
        help="Path to archive targets .npy (default: output/rf_archive_y.npy)",
    )
    parser.add_argument(
        "--output", type=str, default="output/rf_report.png",
        help="Output image path (default: output/rf_report.png)",
    )
    args = parser.parse_args()

    model = _load_rf(args.model)
    X = _load_npy(args.X)
    y = _load_npy(args.y)

    plot_rf_report(model, X, y, output=args.output)


if __name__ == "__main__":
    main()
