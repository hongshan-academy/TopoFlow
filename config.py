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
    max_edges: int = 60
    internal_nodes_choices: List[int] = field(default_factory=lambda: list(range(10, 51)))
    internal_nodes_weights: List[float] = field(default_factory=lambda: [
        math.exp(-((x - 30) ** 2) / (2 * 8 ** 2)) for x in range(10, 51)
    ])
    max_generation_tries: int = 500
    balanced_config_tries: int = 2000
    chromosome_retries: int = 2000        # max tries in make_random_chromosome
    individual_retries: int = 100         # max tries in _generate_random_individual

    # ── GA ──
    pop_size: int = 80
    generations: int = 2000
    target_pq: Tuple[int, int] = (325, 799)
    mutation_rate: float = 0.85
    crossover_rate: float = 0.35
    tournament_size: int = 3
    elitism_count: int = 1
    immigration_rate: float = 0.12
    eval_timeout: Optional[float] = None

    # ── Stagnation / Restart ──
    stagnation_interval: int = 15        # boost immigration every N stagnant gens
    stagnation_restart: int = 30         # full restart after N stagnant gens
    stagnation_boost_ratio: float = 0.7  # fraction of pop_size for boost immigration
    restart_survivors: int = 2           # elites kept on full restart

    # ── Permutation encoding ──
    balance_max_depth: int = 10
    repair_max_iter_factor: int = 2
    repair_retry_on_fail: bool = True

    # ── GA operators ──
    mutation_max_tries: int = 5
    mutation_weights: List[float] = field(default_factory=lambda: [
        0.16,  # mutate_counts (explore)
        0.11,  # mutate_counts_preserve
        0.18,  # mutate_perm_swap
        0.22,  # mutate_perm_scramble
        0.23,  # mutate_perm_reverse
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

    # ── Output ──
    output_path: str = "output/ga_top5.json"
    history_path: str = "output/ga_history.json"


DEFAULT_CONFIG = Config()
