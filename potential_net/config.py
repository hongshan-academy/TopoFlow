from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class PotentialNetConfig:
    # ── Model ──
    conv_type: Literal["GCN", "GAT"] = "GCN"
    hidden_dim: int = 64
    num_layers: int = 3
    output_sigmoid: bool = True
    hand_dim: int = 0

    # ── Training ──
    learning_rate: float = 3e-4
    epochs: int = 300
    batch_size: int = 256
    early_stop_patience: int = 30
    ranking_margin: float = 0.05
    head_short_weight: float = 0.4
    head_medium_weight: float = 0.6
    reg_weight: float = 0.1
    flow_weight: float = 0.0
    diff_boost: float = 30
    diff_thresh: float = 1e-4

    # ── Data collection ──
    n_targets: int = 500
    collection_pop_size: int = 30
    collection_generations: int = 80
    collection_workers: int = 8
    collection_solver_threads: int = 1
    collection_surrogate_eval_rate: float = 0.05
    collection_random_eval_rate: float = 0.02
    collection_mode: Literal["MILP", "simulation", "mixed"] = "simulation"
    collection_sim_max_frames: int = 5000
    short_horizon: int = 5
    medium_horizon: int = 20
    min_samples_per_target: int = 200

    # ── Mutation budget (GA integration) ──
    mutation_budget_base: int = 1
    mutation_budget_max_extra: int = 4
    budget_short_weight: float = 0.4
    budget_medium_weight: float = 0.6

    # ── Output ──
    model_path: str = "output/potential_net.pt"
    samples_path: str = "output/potential_samples.pt"
    checkpoint_dir: str = "output/potential_checkpoints"


DEFAULT_CONFIG = PotentialNetConfig()
