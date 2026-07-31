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
    seed_edges: List[Tuple[str, str]] = field(default_factory=list)

    # ── Mutation limits ──
    max_in_degree: int = 3
    max_out_degree: int = 3
    subplot_min_edges: int = 3
    subplot_complexity_min: int = 3
    subplot_complexity_max: int = 8

    # ── Random graph generation ──
    max_edges: int = 80
    internal_nodes_choices: List[int] = field(default_factory=lambda: list(range(17, 22)))
    internal_nodes_weights: List[float] = field(default_factory=lambda: [
        math.exp(-((x - 19) ** 2) / (2 * 1.5 ** 2)) for x in range(17, 22)
    ])
    max_generation_tries: int = 500
    balanced_config_tries: int = 2000
    chromosome_retries: int = 2000        # max tries in make_random_chromosome
    individual_retries: int = 100         # max tries in _generate_random_individual

    # ── GA ──
    pop_size: int = 150
    generations: int = 6000
    target_pq: Tuple[int, int] = (325, 799)
    mutation_rate: float = 0.75
    crossover_rate: float = 0.60
    tournament_size: int = 2
    elitism_count: int = 8
    immigration_rate: float = 0.18
    eval_timeout: Optional[float] = None

    # ── Stagnation / Restart ──
    stagnation_interval: int = 20        # boost immigration every N stagnant gens
    stagnation_restart: int = 150        # full restart after N stagnant gens
    stagnation_boost_ratio: float = 0.5  # fraction of pop_size for boost immigration
    restart_survivors: int = 5           # elites kept on full restart

    # ── Permutation encoding ──
    balance_max_depth: int = 10
    repair_max_iter_factor: int = 2
    repair_retry_on_fail: bool = True

    # ── GA operators ──
    mutation_max_tries: int = 5
    mutation_weights: List[float] = field(default_factory=lambda: [
        0.12,  # mutate_counts (explore)
        0.12,  # mutate_counts_preserve
        0.20,  # mutate_perm_swap
        0.20,  # mutate_perm_scramble
        0.26,  # mutate_perm_reverse
        0.10,  # mutate_graph_delete
    ])

    # ── Solver ──
    mode: Literal['MILP', 'simulation', 'mixed'] = 'mixed'
    mixed_edge_threshold: int = 60
    sim_max_frames: Optional[int] = 100000
    solver_cache_size: int = 100000
    solver_threads: int = 1
    solver_workers: int = 16
    infeasible_throughput: float = 0.0   # 0.0 = reject all infeasible; 1.0 = allow all

    # ── Surrogate RF ──
    surrogate_enabled: bool = True
    surrogate_top_fraction: float = 0.25
    surrogate_random_eval_fraction: float = 0.05
    surrogate_warmup_samples: int = 80
    surrogate_retrain_interval: int = 5
    surrogate_n_estimators: int = 150
    surrogate_max_depth: int = 15

    # ── Output ──
    output_path: str = "output/ga_top5.json"
    history_path: str = "output/ga_history.json"


DEFAULT_CONFIG = Config()
