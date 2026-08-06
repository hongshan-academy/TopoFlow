"""
Train the TargetConditionalGNN (Evolutionary Potential Network) on collected
trajectory data using pairwise ranking loss.

Usage:
    python -m potential_net.train
    python -m potential_net.train --samples output/potential_samples.pkl --epochs 300
"""

import argparse
import os
import pickle
import sys
import time
from collections import Counter
from typing import List

import numpy as np
from scipy.stats import spearmanr

from potential_net.config import DEFAULT_CONFIG
from potential_net.data import PotentialDataset
from potential_net.model import PotentialNet


def compute_sample_weights(
    samples: List,
    diff_boost: float,
    diff_thresh: float,
) -> np.ndarray:
    target_counts = Counter(s[1] for s in samples)
    weights = np.empty(len(samples), dtype=np.float64)
    for i, s in enumerate(samples):
        w = 1.0 / target_counts[s[1]]
        if s[3] - s[2] > diff_thresh:
            w *= diff_boost
        weights[i] = w
    mean_w = weights.mean()
    if mean_w > 0:
        weights /= mean_w
    return weights


def load_samples(path: str) -> List:
    print(f"Loading samples from: {path}")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    print(f"  Loaded {len(samples):,} samples")
    return samples


def compute_metrics(
    model: PotentialNet,
    data_list: list,
    y_short: np.ndarray,
    y_medium: np.ndarray,
) -> dict:
    _, s_short, s_medium = model.predict(data_list)
    rho_short, _ = spearmanr(s_short, y_short)
    rho_medium, _ = spearmanr(s_medium, y_medium)
    return {
        "spearman_short": float(rho_short),
        "spearman_medium": float(rho_medium),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Evolutionary Potential Network")
    parser.add_argument("--samples", type=str, default="output/potential_samples.pkl")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG.early_stop_patience)
    parser.add_argument("--margin", type=float, default=DEFAULT_CONFIG.ranking_margin)
    parser.add_argument("--short-weight", type=float, default=DEFAULT_CONFIG.head_short_weight)
    parser.add_argument("--medium-weight", type=float, default=DEFAULT_CONFIG.head_medium_weight)
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CONFIG.checkpoint_dir)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--diff-boost", type=float, default=30.0,
                        help="Sampling boost for samples where medium>short by >--diff-thresh")
    parser.add_argument("--diff-thresh", type=float, default=1e-4,
                        help="Threshold for differentiating samples (medium>short)")
    parser.add_argument("--reg-weight", type=float, default=1.0,
                        help="Weight for regression MSE auxiliary loss (0 = ranking only)")
    parser.add_argument("--sharpen-epsilon", type=float, default=None,
                        help="If set, transform potential labels: exp(-(1-pot)/epsilon) for exact-reachability focus")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "bce"],
                        help="Regression loss type (bce for imbalanced sharp labels)")
    parser.add_argument("--flow-weight", type=float, default=1.0,
                        help="Weight for flow ratio MSE loss (0 = no flow head training)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("  TopoFlow Potential Net — Training")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  patience={args.patience}  margin={args.margin}")
    print(f"  weights: short={args.short_weight}  medium={args.medium_weight}")
    print("=" * 60)

    samples = load_samples(args.samples)
    if not samples:
        print("No samples found. Run collect_trajectories.py first.")
        sys.exit(1)

    if args.sharpen_epsilon is not None and args.sharpen_epsilon > 0:
        eps = args.sharpen_epsilon
        old_s_mean = np.mean([s[2] for s in samples])
        old_m_mean = np.mean([s[3] for s in samples])
        samples = [
            (s[0], s[1],
             float(np.exp(-max(0.0, 1.0 - s[2]) / eps)),
             float(np.exp(-max(0.0, 1.0 - s[3]) / eps)),
             *s[4:])
            for s in samples
        ]
        new_s_mean = np.mean([s[2] for s in samples])
        new_m_mean = np.mean([s[3] for s in samples])
        n_positive = int(sum(1 for s in samples if s[2] > 0.01))
        print(f"  sharpen ε={eps:.0e}: old mean ({old_s_mean:.4f}, {old_m_mean:.4f})"
              f" -> new mean ({new_s_mean:.4f}, {new_m_mean:.4f})"
              f"  |  {n_positive} with pot>0.01 ({n_positive/len(samples)*100:.2f}%)")

    n_total = len(samples)
    indices = np.random.permutation(n_total)
    n_val = max(8, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:]]

    print(f"Train: {n_train:,}  Val: {n_val:,}")

    train_data_list = [PotentialDataset([s])[0] for s in train_samples]
    val_data_list = [PotentialDataset([s])[0] for s in val_samples]

    model = PotentialNet(
        conv_type=DEFAULT_CONFIG.conv_type,
        hidden_dim=DEFAULT_CONFIG.hidden_dim,
        num_layers=DEFAULT_CONFIG.num_layers,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.patience,
        output_sigmoid=DEFAULT_CONFIG.output_sigmoid,
    )

    y_short_train = np.array([s[2] for s in train_samples], dtype=np.float64)
    y_medium_train = np.array([s[3] for s in train_samples], dtype=np.float64)
    y_short_val = np.array([s[2] for s in val_samples], dtype=np.float64)
    y_medium_val = np.array([s[3] for s in val_samples], dtype=np.float64)
    y_flow_train = np.array([s[6] if len(s) > 6 else 0.0 for s in train_samples], dtype=np.float64)
    has_flow = any(len(s) > 6 for s in train_samples)

    print(f"\nTraining...")
    t_start = time.perf_counter()

    sample_weights = compute_sample_weights(train_samples, args.diff_boost, args.diff_thresh)
    n_diff = int((np.array([s[3] for s in train_samples]) - np.array([s[2] for s in train_samples]) > args.diff_thresh).sum())
    print(f"  sample weights: mean=1.0  differentiating samples={n_diff} ({n_diff/len(train_samples)*100:.2f}%)")
    if args.diff_boost != 1.0:
        print(f"  diff_boost={args.diff_boost}x")

    model.fit(
        data_list=train_data_list,
        y_short=y_short_train,
        y_medium=y_medium_train,
        margin=args.margin,
        head_short_weight=args.short_weight,
        head_medium_weight=args.medium_weight,
        sample_weights=sample_weights,
        reg_weight=args.reg_weight,
        loss_type=args.loss_type,
        y_flow=y_flow_train if has_flow else None,
        flow_weight=args.flow_weight,
    )

    elapsed = time.perf_counter() - t_start
    print(f"Training complete in {elapsed:.1f}s")

    print("\nFinal metrics:")
    metrics = compute_metrics(model, val_data_list, y_short_val, y_medium_val)
    print(f"  Spearman ρ (short):  {metrics['spearman_short']:.4f}")
    print(f"  Spearman ρ (medium): {metrics['spearman_medium']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nModel saved to: {args.output}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "potential_net_final.pt")
    model.save(ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
