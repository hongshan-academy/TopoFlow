"""
Standalone GNN pre-training script: generates random graphs, evaluates flow ratios
in parallel, and trains a GNN to predict flow_ratio ∈ [0,1] from graph topology.

Usage:
    python tools/train_gnn.py
    python tools/train_gnn.py --n-samples 15000 --workers 18 --output output/trained_gnn.pt
    python tools/train_gnn.py --target-pq 325 799 --n-samples 5000 --gnn-epochs 300
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from functools import lru_cache
from typing import Any, List, Tuple

import numpy as np

_tools_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tools_dir)
if _tools_dir in sys.path:
    sys.path.remove(_tools_dir)
sys.path.insert(0, _project_root)

_eval_ratio_fn: Any = None


def _evaluate_flow_ratio(
    edges_tuple: Tuple[Tuple[str, str], ...],
    mode: str = "mixed",
    threads: int = 1,
    max_denominator: int = 10000,
) -> float:
    from graph import Graph
    from solver import solve
    from simulator import simulate
    from config import DEFAULT_CONFIG as _cfg

    graph = Graph.from_edges(list(edges_tuple))
    if mode == "mixed":
        actual_mode: str = "simulation" if len(graph.edges) <= _cfg.mixed_edge_threshold else "MILP"
    else:
        actual_mode = mode

    if actual_mode == "simulation":
        sim_result = simulate(graph, max_frames=_cfg.sim_max_frames)
        if not sim_result.converged:
            result = solve(graph, threads=threads)
        else:
            result = sim_result
    else:
        result = solve(graph, threads=threads)

    source = next(iter(graph.sources))
    for edge_result in result.edges:
        if edge_result.source == source:
            frac = Fraction(edge_result.flow).limit_denominator(max_denominator)
            return float(frac)
    return 0.0


def _worker_init_ratio(mode: str, threads: int, max_denominator: int) -> None:
    global _eval_ratio_fn
    from config import DEFAULT_CONFIG as _cfg

    @lru_cache(maxsize=_cfg.solver_cache_size)
    def _cached(edges_tuple: Tuple[Tuple[str, str], ...]) -> float:
        return _evaluate_flow_ratio(edges_tuple, mode=mode, threads=threads, max_denominator=max_denominator)

    _eval_ratio_fn = _cached


def _eval_one_ratio(edges_tuple: Tuple[Tuple[str, str], ...]) -> float:
    global _eval_ratio_fn
    try:
        if _eval_ratio_fn is not None:
            return _eval_ratio_fn(edges_tuple)
        return float("inf")
    except Exception:
        return float("inf")


def generate_graphs(
    n: int,
) -> List[Tuple[Tuple[str, str], ...]]:
    import random
    from ga.generation import generate_strict_graph

    result: List[Tuple[Tuple[str, str], ...]] = []
    tries = 0
    max_tries = n * 2

    print(f"Generating {n:,} random graphs (n_internal 10-20)...")
    while len(result) < n and tries < max_tries:
        tries += 1
        n_internal = random.randint(10, 20)
        g = generate_strict_graph(n_internal)
        if g.is_valid(strict=True):
            result.append(tuple(sorted(g.edges)))
        if len(result) % 1000 == 0 and len(result) > 0:
            print(f"  generated {len(result):,} / {n:,}  (attempts: {tries:,})")

    print(f"  done: {len(result):,} valid graphs from {tries:,} attempts")
    return result


def train_step(
    gnn: Any,
    archive: List[Tuple[Tuple[Tuple[str, str], ...], float]],
    epochs: int,
    batch_size: int,
    cached_data: list | None = None,
) -> list | None:
    from nn_surrogate.data import _edges_to_pyg_data

    edges = [s[0] for s in archive]
    y_arr = np.array([s[1] for s in archive], dtype=np.float64)

    if cached_data is not None and len(cached_data) == len(archive):
        data_list = cached_data
    else:
        data_list = [_edges_to_pyg_data(e, y) for e, y in zip(edges, y_arr)]

    gnn._epochs = epochs
    gnn._batch_size = batch_size
    gnn.fit(edges, y_arr, cached_data=data_list)
    return data_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train a GNN surrogate for flow ratio prediction")
    parser.add_argument("--n-samples", type=int, default=15000,
                        help="Number of graphs to generate and evaluate (default: 15000)")
    parser.add_argument("--workers", type=int, default=18,
                        help="Parallel evaluation workers (default: 18)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Solver threads per worker (default: 1)")
    parser.add_argument("--mode", type=str, default="mixed",
                        choices=["MILP", "simulation", "mixed"],
                        help="Evaluation mode (default: mixed)")
    parser.add_argument("--max-denominator", type=int, default=10000,
                        help="Fraction denominator limit (default: 10000)")
    parser.add_argument("--gnn-conv", type=str, default="GCN", choices=["GCN", "GAT"],
                        help="GNN convolution type (default: GCN)")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64,
                        help="GNN hidden dimension (default: 64)")
    parser.add_argument("--gnn-num-layers", type=int, default=3,
                        help="GNN message-passing layers (default: 3)")
    parser.add_argument("--gnn-lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--gnn-epochs", type=int, default=200,
                        help="Training epochs for final model (default: 200)")
    parser.add_argument("--gnn-inc-epochs", type=int, default=50,
                        help="Training epochs per incremental step (default: 50)")
    parser.add_argument("--gnn-batch-size", type=int, default=256,
                        help="Training batch size (default: 32)")
    parser.add_argument("--gnn-patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--train-every", type=int, default=500,
                        help="Train GNN every N new valid samples (default: 500)")
    parser.add_argument("--checkpoint-every", type=int, default=3000,
                        help="Save checkpoint every N samples (default: 3000)")
    parser.add_argument("--train-min", type=int, default=500,
                        help="Minimum samples before first training (default: 500)")
    parser.add_argument("--output", type=str, default="output/trained_gnn.pt",
                        help="Output model path (default: output/trained_gnn.pt)")
    parser.add_argument("--output-samples", type=str, default="output/gnn_samples.pkl",
                        help="Output samples pickle path (default: output/gnn_samples.pkl)")
    parser.add_argument("--ckpt-dir", type=str, default="output/gnn_checkpoints",
                        help="Checkpoint directory (default: output/gnn_checkpoints)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable intermediate checkpoints")
    args = parser.parse_args()

    from nn_surrogate.model import SurrogateGNN

    print("=" * 60)
    print("  TopoFlow GNN Pre-Training")
    print(f"  n_samples={args.n_samples:,}  workers={args.workers}  mode={args.mode}")
    print(f"  GNN: {args.gnn_conv}(h={args.gnn_hidden_dim}, L={args.gnn_num_layers})")
    print(f"  epochs={args.gnn_epochs}  inc_epochs={args.gnn_inc_epochs}  batch={args.gnn_batch_size}")
    print(f"  train_every={args.train_every}  checkpoint_every={args.checkpoint_every}")
    print("=" * 60)

    all_edges = generate_graphs(args.n_samples)
    if not all_edges:
        print("No valid graphs generated. Exiting.")
        sys.exit(1)

    gnn = SurrogateGNN(
        conv_type=args.gnn_conv,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.gnn_num_layers,
        learning_rate=args.gnn_lr,
        epochs=args.gnn_epochs,
        batch_size=args.gnn_batch_size,
        early_stop_patience=args.gnn_patience,
        output_sigmoid=True,
    )

    archive: List[Tuple[Tuple[Tuple[str, str], ...], float]] = []
    total_invalid = 0
    total_evaluated = 0
    t_start = time.perf_counter()

    print(f"\nEvaluating {len(all_edges):,} graphs with {args.workers} workers...")
    print(f"  Incremental training every {args.train_every} samples "
          f"(min {args.train_min} to start)")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    cached_data = None

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init_ratio,
        initargs=(args.mode, args.threads, args.max_denominator),
    ) as executor:
        futures = {
            executor.submit(_eval_one_ratio, edges): edges
            for edges in all_edges
        }

        for fut in as_completed(futures):
            total_evaluated += 1
            ratio = fut.result(timeout=60)

            if ratio != float("inf") and ratio >= 0.0:
                edges = futures[fut]
                archive.append((edges, ratio))
            else:
                total_invalid += 1

            n_valid = len(archive)
            if n_valid % args.train_every == 0 and n_valid >= args.train_min:
                t0 = time.perf_counter()
                cached_data = train_step(gnn, archive, epochs=args.gnn_inc_epochs,
                                         batch_size=args.gnn_batch_size, cached_data=cached_data)
                dt = time.perf_counter() - t0
                print(f"\n  [{n_valid:,} samples] GNN trained in {dt:.1f}s")

            if not args.no_checkpoints and n_valid > 0 and n_valid % args.checkpoint_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"gnn_{n_valid}.pt")
                gnn.save(ckpt_path)
                print(f"  checkpoint saved: {ckpt_path}")

            if total_evaluated % 2000 == 0:
                elapsed = time.perf_counter() - t_start
                rate = total_evaluated / elapsed if elapsed > 0 else 0
                print(f"\n  progress: {total_evaluated:,} evaluated  "
                      f"({n_valid:,} valid, {total_invalid:,} invalid)  "
                      f"{rate:.0f} eval/s")

    t_collect = time.perf_counter() - t_start
    print(f"\nEvaluation complete in {t_collect:.1f}s")
    print(f"  Valid: {len(archive):,}  Invalid: {total_invalid:,}")

    if len(archive) < args.train_min:
        print(f"Not enough valid samples ({len(archive)} < {args.train_min}) to train. Exiting.")
        sys.exit(1)

    print(f"\nFinal training on {len(archive):,} samples ({args.gnn_epochs} epochs)...")
    t_train = time.perf_counter()
    train_step(gnn, archive, epochs=args.gnn_epochs, batch_size=args.gnn_batch_size, cached_data=cached_data)
    dt_train = time.perf_counter() - t_train
    print(f"  trained in {dt_train:.1f}s")

    gnn.save(args.output)
    print(f"Model saved to: {args.output}")

    import pickle
    with open(args.output_samples, "wb") as f:
        pickle.dump(archive, f)
    print(f"Samples saved to: {args.output_samples}")

    from rf_surrogate.features import extract_features
    X_arr = np.array([extract_features(s[0]) for s in archive], dtype=np.float64)
    y_arr = np.array([s[1] for s in archive], dtype=np.float64)
    base = os.path.splitext(args.output_samples)[0]
    np.save(f"{base}_X.npy", X_arr)
    np.save(f"{base}_y.npy", y_arr)
    print(f"Feature arrays saved to: {base}_X.npy, {base}_y.npy")

    print(f"\nDone. Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"\nTo use in GA:")
    print(f"  from ga.core import run")
    print(f"  run(surrogate_type='gnn')")
    print(f"  (will auto-load {args.output} if present)")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
