from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional

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
    extended_patterns: Set[Tuple[int, int]] = field(default_factory=lambda: {
        (1, 2), (1, 3), (2, 1), (3, 1), (0, 1), (1, 0),
        (1, 1), (0, 0), (2, 2),
    })
    seed_edges: List[Tuple[str, str]] = field(default_factory=list)
    crossing_node_fmt: str = "_X_{count}_{e1}_{e2}"

    # ── Mutation limits ──
    max_in_degree: int = 3
    max_out_degree: int = 3
    subplot_min_edges: int = 3
    subplot_complexity_min: int = 3
    subplot_complexity_max: int = 8
    subplot_max_crossings: int = 1

    # ── Random graph generation ──
    # internal_nodes_choices: List[int] = field(default_factory=lambda: [24])
    # internal_nodes_weights: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 2, 3, 4, 4, 4 , 3 , 2 ])
    # internal_nodes_weights: List[int] = field(default_factory=lambda: [1])
    internal_nodes_choices: List[int] = field(default_factory=lambda: list(range(25, 46)))
    internal_nodes_weights: List[float] = field(default_factory=lambda: [
        math.exp(-((x - 35) ** 2) / (2 * 5 ** 2)) for x in range(25, 46)
    ])
    max_generation_tries: int = 500
    balanced_config_tries: int = 2000
    random_crossing_prob: float = 0.3

    # ── GA ──
    pop_size: int = 80
    generations: int = 2000
    # target_f: float = 325 / 799
    target_pq: Tuple[int, int] = (325, 799)
    mutation_rate: float = 0.5
    crossover_rate: float = 0.5
    tournament_size: int = 2
    elitism_count: int = 2
    immigration_rate: float = 0.05
    eval_timeout: Optional[float] = None

    # ── Solver ──
    solver_cache_size: int = 10000
    solver_threads: int = 1
    solver_workers: int = 16
    infeasible_throughput: float = 0.0

    # ── Output ──
    output_path: str = "output/ga_top5.json"


DEFAULT_CONFIG = Config()
