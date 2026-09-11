from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Tuple, Optional


@dataclass
class Config:
    # -- Graph --
    strict_patterns: Set[Tuple[int, int]] = field(default_factory=lambda: {
        (1, 2), (1, 3), (2, 1), (3, 1), (0, 1), (1, 0),
    })
    patterns: Set[Tuple[int, int]] = field(default_factory=lambda: {
        (1, 2), (1, 3), (2, 1), (3, 1), (0, 1), (1, 0),
        (1, 1), (0, 0),
    })

    # -- Simulator --
    sim_max_frames: Optional[int] = 5000


DEFAULT_CONFIG = Config()
