import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GenStats:
    gen: int
    best_error: float
    best_nodes: int
    avg_error: float
    min_nodes: int
    max_nodes: int
    max_edges: int
    n_mutated: int
    n_selected: int
    n_immigrants: int
    n_ready: int
    n_pending: int
    n_harvested: int
    elapsed_sec: float
    n_crossed: int = 0
    best_edges: Optional[List[List[str]]] = None


@dataclass
class BestSnapshot:
    gen: int
    edges: List[List[str]]
    fitness: Tuple[float, int]
    graph_nodes: int
    graph_edges: int
    uid: Optional[str] = None


class History:
    def __init__(self, record_population: bool = False):
        self.start_time = time.perf_counter()
        self.elapsed_sec: float = 0.0
        self.target_pq: Optional[Tuple[int, int]] = None
        self.params: Dict[str, Any] = {}
        self.generations: List[GenStats] = []
        self.best_chain: List[BestSnapshot] = []
        self._record_population = record_population
        self.population_snapshots: List[Dict[str, Any]] = []

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def record_gen(
        self,
        gen: int,
        best_error: float,
        best_nodes: int,
        avg_error: float,
        min_nodes: int,
        max_nodes: int,
        max_edges: int,
        n_crossed: int = 0,
        n_mutated: int = 0,
        n_selected: int = 0,
        n_immigrants: int = 0,
        n_ready: int = 0,
        n_pending: int = 0,
        n_harvested: int = 0,
        elapsed_sec: float = 0.0,
        best_edges: Optional[List[List[str]]] = None,
    ) -> None:
        self.generations.append(GenStats(
            gen=gen,
            best_error=best_error,
            best_nodes=best_nodes,
            avg_error=avg_error,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_edges=max_edges,
            n_crossed=n_crossed,
            n_mutated=n_mutated,
            n_selected=n_selected,
            n_immigrants=n_immigrants,
            n_ready=n_ready,
            n_pending=n_pending,
            n_harvested=n_harvested,
            elapsed_sec=elapsed_sec,
            best_edges=best_edges,
        ))

    def record_best(
        self,
        gen: int,
        edges: List[List[str]],
        fitness: Tuple[float, int],
        graph_nodes: int,
        graph_edges: int,
        uid: Optional[str] = None,
    ) -> None:
        if self.best_chain:
            prev = self.best_chain[-1]
            if prev.fitness == fitness:
                return
        self.best_chain.append(BestSnapshot(
            gen=gen,
            edges=edges,
            fitness=fitness,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            uid=uid,
        ))

    def record_population(self, gen: int, individuals: List[Dict[str, Any]]) -> None:
        if self._record_population:
            self.population_snapshots.append({
                "gen": gen,
                "individuals": individuals,
            })

    def to_dict(self) -> Dict[str, Any]:
        self.elapsed_sec = time.perf_counter() - self.start_time
        d: Dict[str, Any] = {
            "target": {
                "p": self.target_pq[0] if self.target_pq else None,
                "q": self.target_pq[1] if self.target_pq else None,
            },
            "params": self.params,
            "elapsed_sec": self.elapsed_sec,
            "generations": [asdict(s) for s in self.generations],
            "best_chain": [asdict(s) for s in self.best_chain],
        }
        if self._record_population:
            d["population_snapshots"] = self.population_snapshots
        return d

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"History exported to: {path}  ({len(self.generations)} generations, "
              f"{len(self.best_chain)} best-chain steps)")

    def summary(self) -> str:
        if not self.generations:
            return "History: no data recorded."
        first = self.generations[0]
        last = self.generations[-1]
        lines = [
            f"History Summary",
            f"  Generations:       {len(self.generations)}",
            f"  Best chain steps:  {len(self.best_chain)}",
            f"  Initial best:      err={first.best_error:.6f} nodes={first.best_nodes}",
            f"  Final best:        err={last.best_error:.6f} nodes={last.best_nodes}",
            f"  Total elapsed:     {self.elapsed_sec:.1f}s",
        ]
        return "\n".join(lines)
