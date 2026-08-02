from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional, Literal

import math


@dataclass
class Config:
    # ── Graph ──
    strict_patterns: Set[Tuple[int, int]] = field(default_factory=lambda: {
        (1, 2), (1, 3), (2, 1), (3, 1), (0, 1), (1, 0),
    })
    patterns: Set[Tuple[int, int]] = field(default_factory=lambda: {
        (1, 2), (1, 3), (2, 1), (3, 1), (0, 1), (1, 0),
        (1, 1), (0, 0),
    })
    seed_path: Optional[str] = "data/seed_graphs.json"

    # ── Mutation limits ──
    max_in_degree: int = 3
    max_out_degree: int = 3
    subplot_min_edges: int = 3
    subplot_complexity_min: int = 3
    subplot_complexity_max: int = 8

    # ── Random graph generation ──
    max_edges: int = 50
    internal_nodes_choices: List[int] = field(default_factory=lambda: list(range(10, 61)))
    internal_nodes_weights: List[float] = field(default_factory=lambda: [
        math.exp(-((x - 35) ** 2) / (2 * 1.5 ** 2)) for x in range(10, 61)
    ])
    max_generation_tries: int = 500
    balanced_config_tries: int = 2000
    individual_retries: int = 100         # max tries in _generate_random_individual

    # ── GA ──
    pop_size: int = 150
    generations: int = 500
    target_pq: Tuple[int, int] = (325, 799)
    mutation_rate: float = 0.68
    crossover_rate: float = 0.55
    tournament_size: int = 3
    elitism_count: int = 12
    immigration_rate: float = 0.22
    eval_timeout: Optional[float] = None

    # ── Stagnation / Restart ──
    stagnation_interval: int = 10        # boost immigration every N stagnant gens
    stagnation_restart: int = 50         # full restart after N stagnant gens
    stagnation_boost_ratio: float = 0.40 # fraction of pop_size for boost immigration
    restart_survivors: int = 5           # elites kept on full restart

    # ── GA operators ──
    mutation_max_tries: int = 5
    mutation_weights: List[float] = field(default_factory=lambda: [
        0.10,  # mutate_partial_delete
        0.15,  # mutate_add_subgraph
        0.05,  # mutate_replace_subgraph
        0.70,  # mutate_reverse_edge
    ])
    crossover_weights: List[float] = field(default_factory=lambda: [
        0.70,  # crossover_subgraph_exchange
        0.30,  # crossover_concat
    ])

    # ── Solver ──
    mode: Literal['MILP', 'simulation', 'mixed'] = 'mixed'
    mixed_edge_threshold: int = 61
    sim_max_frames: Optional[int] = 100000
    solver_cache_size: int = 100000
    solver_threads: int = 1
    solver_workers: int = 16
    infeasible_throughput: float = 0.0   # 0.0 = reject all infeasible; 1.0 = allow all

    # ── Surrogate RF ──
    surrogate_enabled: bool = True
    surrogate_top_fraction: float = 0.12
    surrogate_random_eval_fraction: float = 0.10
    surrogate_warmup_samples: int = 80
    surrogate_retrain_interval: int = 5
    surrogate_n_estimators: int = 150
    surrogate_max_depth: int = 15

    # ── Output ──
    output_path: str = "output/ga_top5.json"
    history_path: str = "output/ga_history.json"


DEFAULT_CONFIG = Config()
