"""
Standalone trajectory collection: runs GA for many random target ratios,
tracks parent→child lineage, and computes multi-horizon evolutionary potentials.

Usage:
    python -m potential_net.collect_trajectories
    python -m potential_net.collect_trajectories --n-targets 500 --pop-size 30 --generations 80
"""

import argparse
import gc
import os
import sys
import time
import random
import pickle
import concurrent.futures as cf
from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from deap import base, creator, tools

from config import DEFAULT_CONFIG as _cfg
from graph import Graph, Edge
from ga.generation import generate_strict_graph
from ga.mutation import MUTATION_FNS
from ga.crossover import CROSSOVER_FNS

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", tuple, fitness=creator.FitnessMin)

_worker_eval_fn: Any = None


def _worker_init(target_pq: Tuple[int, int], threads: int, max_denominator: int = 10000,
                 mode: str = "mixed", sim_max_frames: Optional[int] = None) -> None:
    global _worker_eval_fn
    if sim_max_frames is not None:
        from config import DEFAULT_CONFIG as _cfg_w
        _cfg_w.sim_max_frames = sim_max_frames
    from ga.fitness import make_evaluate
    _worker_eval_fn = make_evaluate(target_pq, threads, max_denominator=max_denominator, mode=mode)  # type: ignore[arg-type]


def _eval_one(edges_tuple: Tuple[Edge, ...]) -> Tuple[float, int, float]:
    global _worker_eval_fn
    try:
        if _worker_eval_fn is not None:
            err, nodes, flow = _worker_eval_fn(edges_tuple)
            return (err, nodes, flow)
        return (float("inf"), 0, 0.0)
    except Exception:
        return (float("inf"), 0, 0.0)


def _generate_random_individual() -> creator.Individual:
    for _ in range(_cfg.individual_retries):
        n_internal = random.choices(
            _cfg.internal_nodes_choices,
            weights=_cfg.internal_nodes_weights,
            k=1,
        )[0]
        g = generate_strict_graph(n_internal)
        if g.is_valid(strict=True) and len(g.edges) <= _cfg.max_edges:
            return creator.Individual(tuple(sorted(g.edges)))
    n_internal = min(_cfg.internal_nodes_choices)
    g = generate_strict_graph(n_internal)
    return creator.Individual(tuple(sorted(g.edges)))


def _mutate_wrapper(
    individual: creator.Individual,
    weights: Tuple[float, ...],
    max_tries: int = 5,
) -> Tuple[creator.Individual]:
    for _ in range(max_tries):
        fn = random.choices(MUTATION_FNS, weights=weights, k=1)[0]
        g = Graph.from_edges(list(individual))
        result = fn(g)
        if result is not None and len(result.edges) <= _cfg.max_edges:
            return (creator.Individual(tuple(sorted(result.edges))),)
    return (individual,)


def _crossover_wrapper(
    ind1: creator.Individual,
    ind2: creator.Individual,
    weights: Tuple[float, ...],
) -> Tuple[creator.Individual, creator.Individual]:
    g1 = Graph.from_edges(list(ind1))
    g2 = Graph.from_edges(list(ind2))
    fn = random.choices(CROSSOVER_FNS, weights=weights, k=1)[0]
    c1, c2 = fn(g1, g2)
    if len(c1.edges) > _cfg.max_edges or len(c2.edges) > _cfg.max_edges:
        return (ind1, ind2)
    return (
        creator.Individual(tuple(sorted(c1.edges))),
        creator.Individual(tuple(sorted(c2.edges))),
    )


class LineageTracker:
    def __init__(self) -> None:
        self.parent_to_children: Dict[int, List[int]] = {}
        self.ind_gen: Dict[int, int] = {}
        self.ind_fitness: Dict[int, Tuple[float, int]] = {}
        self.ind_flow: Dict[int, float] = {}
        self.ind_edges: Dict[int, Tuple[Edge, ...]] = {}

    def record_birth(self, ind_id: int, edges: Tuple[Edge, ...], gen: int) -> None:
        if ind_id not in self.ind_gen:
            self.ind_gen[ind_id] = gen
        if ind_id not in self.ind_edges:
            self.ind_edges[ind_id] = edges

    def record_parent_child(self, parent_id: int, child_id: int) -> None:
        self.parent_to_children.setdefault(parent_id, []).append(child_id)

    def record_fitness(self, ind_id: int, fitness: Tuple[float, int], flow_ratio: float = 0.0) -> None:
        self.ind_fitness[ind_id] = fitness
        self.ind_flow[ind_id] = flow_ratio

    def compute_potentials(
        self,
        horizon: int,
        epsilon: Optional[float] = None,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        children_map: Dict[int, Set[int]] = {}
        for p_id, c_ids in self.parent_to_children.items():
            children_map.setdefault(p_id, set()).update(c_ids)

        best_errors: Dict[int, float] = {}
        potentials: Dict[int, float] = {}
        for ind_id in self.ind_gen:
            best_error = float("inf")
            queue: deque = deque([(ind_id, 0)])
            visited: Set[int] = {ind_id}

            while queue:
                current, depth = queue.popleft()
                if depth >= horizon:
                    continue
                for child_id in children_map.get(current, set()):
                    if child_id not in visited:
                        visited.add(child_id)
                        err, _ = self.ind_fitness.get(child_id, (float("inf"), 0))
                        if err < best_error:
                            best_error = err
                        queue.append((child_id, depth + 1))

            if best_error == float("inf") and ind_id in self.ind_fitness:
                best_error = self.ind_fitness[ind_id][0]

            best_errors[ind_id] = 1.0 if best_error == float("inf") else best_error
            if best_error == float("inf"):
                potentials[ind_id] = 0.0
            elif epsilon is not None and epsilon > 0:
                potentials[ind_id] = float(np.exp(-best_error / epsilon))
            else:
                potentials[ind_id] = 1.0 - best_error

        return potentials, best_errors

    def get_labeled_samples(
        self,
        target_pq: Tuple[int, int],
        short_horizon: int = 5,
        medium_horizon: int = 20,
    ) -> List[Tuple[Tuple[Edge, ...], Tuple[int, int], float, float, float, float, float]]:
        p_short, e_short = self.compute_potentials(short_horizon)
        p_medium, e_medium = self.compute_potentials(medium_horizon)

        samples: List = []
        for ind_id in self.ind_fitness:
            if ind_id not in self.ind_edges:
                continue
            fitness_err = self.ind_fitness[ind_id][0]
            if fitness_err == float("inf"):
                continue
            samples.append((
                self.ind_edges[ind_id],
                target_pq,
                p_short.get(ind_id, 0.0),
                p_medium.get(ind_id, 0.0),
                e_short.get(ind_id, 1.0),
                e_medium.get(ind_id, 1.0),
                self.ind_flow.get(ind_id, 0.0),
            ))
        return samples


def _evolve_with_lineage(
    population: List[creator.Individual],
    toolbox: base.Toolbox,
    ngen: int,
    mutation_rate: float,
    crossover_rate: float,
    threads: int,
    executor: cf.ProcessPoolExecutor,
    tracker: LineageTracker,
    elitism_count: int = 0,
    immigration_rate: float = 0.0,
    mutation_weights: Optional[Tuple[float, ...]] = None,
    crossover_weights: Optional[Tuple[float, ...]] = None,
) -> None:
    pop_size = len(population)
    n_workers = executor._max_workers if hasattr(executor, '_max_workers') else 1

    if mutation_weights is None:
        mutation_weights = tuple(_cfg.mutation_weights)
    if crossover_weights is None:
        crossover_weights = tuple(_cfg.crossover_weights)

    _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0

    pending: Dict[cf.Future[Tuple[float, int, float]], int] = {}

    for gen in range(ngen):
        for fut in [f for f in pending if f.done()]:
            idx = pending.pop(fut)
            err, nodes, flow = fut.result()
            population[idx].fitness.values = (err, nodes)
            tracker.record_fitness(id(population[idx]), (err, nodes), flow)

        if pending:
            pending_idxs_set = set(pending.values())
            ready_check = [ind for i, ind in enumerate(population) if i not in pending_idxs_set]
            desired_ready = max(elitism_count + 1, pop_size - n_workers)
            if len(ready_check) < desired_ready:
                while pending:
                    pending_idxs_set = set(pending.values())
                    ready_check = [ind for i, ind in enumerate(population) if i not in pending_idxs_set]
                    if len(ready_check) >= desired_ready:
                        break
                    done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        err, nodes, flow = fut.result()
                        population[idx].fitness.values = (err, nodes)
                        tracker.record_fitness(id(population[idx]), (err, nodes), flow)

        new_pending_idxs: Set[int] = set(pending.values())
        ready = [ind for i, ind in enumerate(population) if i not in new_pending_idxs]

        if not ready and pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                err, nodes, flow = fut.result()
                population[idx].fitness.values = (err, nodes)
                tracker.record_fitness(id(population[idx]), (err, nodes), flow)
            new_pending_idxs = set(pending.values())
            ready = [ind for i, ind in enumerate(population) if i not in new_pending_idxs]

        ready.sort(key=lambda ind: ind.fitness.values)
        elites = []
        for ind in ready[:elitism_count]:
            clone = toolbox.clone(ind)
            tracker.record_parent_child(id(ind), id(clone))
            tracker.record_birth(id(clone), tuple(clone), gen)
            tracker.record_fitness(id(clone), clone.fitness.values, tracker.ind_flow.get(id(ind), 0.0))
            elites.append(clone)

        n_immigrants = _immig_rate
        select_k = pop_size - elitism_count - len(new_pending_idxs) - n_immigrants
        if select_k < 0:
            n_immigrants = max(0, pop_size - elitism_count - len(new_pending_idxs))
            select_k = 0

        selected: List[creator.Individual] = []
        if select_k > 0:
            selected = toolbox.select(ready, select_k)
        selected = [toolbox.clone(ind) for ind in selected]

        for i in range(0, len(selected) - 1, 2):
            if random.random() < crossover_rate:
                parent1_id = id(selected[i])
                parent2_id = id(selected[i + 1])
                c1, c2 = _crossover_wrapper(selected[i], selected[i + 1], crossover_weights)
                selected[i] = creator.Individual(tuple(sorted(c1)))
                selected[i + 1] = creator.Individual(tuple(sorted(c2)))
                del selected[i].fitness.values
                del selected[i + 1].fitness.values
                child1_id = id(selected[i])
                child2_id = id(selected[i + 1])
                tracker.record_parent_child(parent1_id, child1_id)
                tracker.record_parent_child(parent1_id, child2_id)
                tracker.record_parent_child(parent2_id, child1_id)
                tracker.record_parent_child(parent2_id, child2_id)
                tracker.record_birth(child1_id, tuple(selected[i]), gen)
                tracker.record_birth(child2_id, tuple(selected[i + 1]), gen)

        for i, ind in enumerate(selected):
            if random.random() < mutation_rate:
                parent_id = id(ind)
                mutated, = _mutate_wrapper(ind, mutation_weights)
                selected[i] = creator.Individual(tuple(sorted(mutated)))
                del selected[i].fitness.values
                child_id = id(selected[i])
                tracker.record_parent_child(parent_id, child_id)
                tracker.record_birth(child_id, tuple(selected[i]), gen)

        immigrants = [toolbox.individual() for _ in range(n_immigrants)]
        for ind in immigrants:
            tracker.record_birth(id(ind), tuple(ind), gen)

        for i, ind in enumerate(selected):
            if len(tuple(ind)) > _cfg.max_edges:
                selected[i] = toolbox.individual()
                tracker.record_birth(id(selected[i]), tuple(selected[i]), gen)
        for i, ind in enumerate(immigrants):
            if len(tuple(ind)) > _cfg.max_edges:
                immigrants[i] = toolbox.individual()
                tracker.record_birth(id(immigrants[i]), tuple(immigrants[i]), gen)

        sorted_pending_idxs = sorted(new_pending_idxs)
        pending_individuals = [population[i] for i in sorted_pending_idxs]

        old_to_new: Dict[int, int] = {}
        for new_i, old_i in enumerate(sorted_pending_idxs):
            old_to_new[old_i] = elitism_count + new_i

        remapped: Dict[cf.Future[Tuple[float, int, float]], int] = {}
        for fut, old_idx in pending.items():
            remapped[fut] = old_to_new[old_idx]
        pending = remapped

        new_pop = elites + pending_individuals + selected + immigrants
        population[:] = new_pop

        offset = elitism_count + len(pending_individuals)
        for i, ind in enumerate(selected + immigrants):
            fut = executor.submit(_eval_one, tuple(ind))
            pending[fut] = offset + i

    while pending:
        done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
        for fut in done:
            idx = pending.pop(fut)
            err, nodes, flow = fut.result()
            population[idx].fitness.values = (err, nodes)
            tracker.record_fitness(id(population[idx]), (err, nodes), flow)


def collect_for_target(
    target_pq: Tuple[int, int],
    pop_size: int = 50,
    generations: int = 100,
    solver_workers: int = 20,
    solver_threads: int = 1,
    mode: str = "mixed",
    mutation_rate: float = 0.68,
    crossover_rate: float = 0.55,
    tournament_size: int = 3,
    elitism_count: int = 3,
    immigration_rate: float = 0.22,
    mutation_weights: Optional[Tuple[float, ...]] = None,
    crossover_weights: Optional[Tuple[float, ...]] = None,
    short_horizon: int = 5,
    medium_horizon: int = 20,
    min_samples: int = 200,
    sim_max_frames: Optional[int] = None,
) -> List[Tuple[Tuple[Edge, ...], Tuple[int, int], float, float]]:
    if pop_size < elitism_count + 1:
        elitism_count = max(0, pop_size - 1)
    if tournament_size > pop_size:
        tournament_size = max(2, pop_size - 1)

    toolbox = base.Toolbox()
    toolbox.register("individual", _generate_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", partial(_crossover_wrapper,
                                     weights=crossover_weights if crossover_weights else tuple(_cfg.crossover_weights)))
    toolbox.register("mutate", partial(_mutate_wrapper,
                                       weights=mutation_weights if mutation_weights else tuple(_cfg.mutation_weights),
                                       max_tries=_cfg.mutation_max_tries))

    population = toolbox.population(n=pop_size)
    tracker = LineageTracker()

    for ind in population:
        tracker.record_birth(id(ind), tuple(ind), 0)

    with cf.ProcessPoolExecutor(
        max_workers=solver_workers,
        initializer=_worker_init,
        initargs=(target_pq, solver_threads, 10000, mode, sim_max_frames),
    ) as executor:
        fut_to_idx = {
            executor.submit(_eval_one, tuple(ind)): i
            for i, ind in enumerate(population)
        }
        done_set, _ = cf.wait(list(fut_to_idx), return_when=cf.ALL_COMPLETED)
        for fut in done_set:
            idx = fut_to_idx[fut]
            err, nodes, flow = fut.result()
            population[idx].fitness.values = (err, nodes)
            tracker.record_fitness(id(population[idx]), (err, nodes), flow)

        replace_idxs = [
            i for i, ind in enumerate(population)
            if ind.fitness.values[0] == float("inf") and random.random() >= _cfg.infeasible_throughput
        ]
        if replace_idxs:
            for i in replace_idxs:
                population[i] = toolbox.individual()
                tracker.record_birth(id(population[i]), tuple(population[i]), 0)
            fut_to_idx = {
                executor.submit(_eval_one, tuple(population[i])): i
                for i in replace_idxs
            }
            done_set, _ = cf.wait(list(fut_to_idx), return_when=cf.ALL_COMPLETED)
            for fut in done_set:
                idx = fut_to_idx[fut]
                err, nodes, flow = fut.result()
                population[idx].fitness.values = (err, nodes)
                tracker.record_fitness(id(population[idx]), (err, nodes), flow)

        _evolve_with_lineage(
            population, toolbox, generations, mutation_rate, crossover_rate,
            solver_threads, executor, tracker,
            elitism_count=elitism_count,
            immigration_rate=immigration_rate,
            mutation_weights=mutation_weights,
            crossover_weights=crossover_weights,
        )

    samples = tracker.get_labeled_samples(
        target_pq,
        short_horizon=short_horizon,
        medium_horizon=medium_horizon,
    )

    if len(samples) == 0:
        return []

    if len(samples) > min_samples * 3:
        indices = np.random.choice(len(samples), size=min_samples * 3, replace=False)
        samples = [samples[i] for i in indices]

    return samples


def _monitor_memory() -> Optional[str]:
    """Return a resource usage summary string, or None if psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1024 ** 3
    try:
        vms_gb = proc.memory_info().vms / 1024 ** 3
    except Exception:
        vms_gb = 0.0
    cpu_pct = proc.cpu_percent(interval=None)
    vm_total = psutil.virtual_memory().total / 1024 ** 3
    vm_used = psutil.virtual_memory().used / 1024 ** 3
    return (f"mem={rss_gb:.1f}GB (vms={vms_gb:.1f}GB) cpu={cpu_pct:.0f}% "
            f"sys_mem={vm_used:.1f}/{vm_total:.1f}GB")


def _sample_target_in_range(lo: float, hi: float) -> Optional[Tuple[int, int]]:
    """Sample a target (p, q) whose ratio p/q is (roughly) uniform in [lo, hi)."""
    for _ in range(200):
        ratio = random.uniform(lo, hi)
        q = random.randint(2, 1000)
        p = int(round(ratio * q))
        if 1 <= p < q and lo <= p / q < hi:
            return (p, q)
    return None


def _build_deficit_targets(
    existing_path: str,
    bucket_target: int,
    ratio_buckets: int,
) -> List[Tuple[int, int]]:
    """Compute per-bucket deficits from existing samples and sample to fill them."""
    with open(existing_path, "rb") as f:
        existing = pickle.load(f)

    counts = [0] * ratio_buckets
    seen: Set[Tuple[int, int]] = set()
    for s in existing:
        p, q = s[1]
        if (p, q) in seen:
            continue
        seen.add((p, q))
        ratio = p / q
        b = min(int(ratio * ratio_buckets), ratio_buckets - 1)
        counts[b] += 1

    deficits = [max(0, bucket_target - c) for c in counts]
    print(f"Existing bucket counts ({ratio_buckets} buckets): {counts}")
    print(f"Deficits (target={bucket_target}/bucket): {deficits}")
    total_deficit = sum(deficits)
    print(f"Total targets to sample: {total_deficit}")

    targets: List[Tuple[int, int]] = []
    for b, d in enumerate(deficits):
        lo = max(0.001, b / ratio_buckets)
        hi = (b + 1) / ratio_buckets
        for _ in range(d):
            for _ in range(500):
                t = _sample_target_in_range(lo, hi)
                if t is not None and t not in seen:
                    seen.add(t)
                    targets.append(t)
                    break
            else:
                print(f"  Warning: could not sample bucket {b} ({lo:.3f}-{hi:.3f})")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GA trajectory data for potential network training")
    parser.add_argument("--n-targets", type=int, default=500, help="Number of target ratios to sample")
    parser.add_argument("--pop-size", type=int, default=30, help="GA population size per target")
    parser.add_argument("--generations", type=int, default=80, help="GA generations per target")
    parser.add_argument("--workers", type=int, default=8, help="Solver workers per GA run (keep low to avoid OOM)")
    parser.add_argument("--solver-threads", type=int, default=1, help="Threads per solver")
    parser.add_argument("--mode", type=str, default="mixed", choices=["MILP", "simulation", "mixed"])
    parser.add_argument("--sim-max-frames", type=int, default=5000,
                        help="Simulation max frames before MILP fallback (lower = faster but less accurate)")
    parser.add_argument("--short-horizon", type=int, default=5)
    parser.add_argument("--medium-horizon", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--mutation-rate", type=float, default=0.68)
    parser.add_argument("--crossover-rate", type=float, default=0.55)
    parser.add_argument("--output", type=str, default="output/potential_samples.pkl")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save partial results every N targets to avoid OOM")
    parser.add_argument("--monitor", action="store_true",
                        help="Print per-target memory/CPU usage via psutil (falls back to os if unavailable)")
    parser.add_argument("--deficit-buckets", action="store_true",
                        help="Sample targets to fill under-populated ratio buckets up to --bucket-target")
    parser.add_argument("--bucket-target", type=int, default=55,
                        help="Reference target count per ratio bucket for --deficit-buckets")
    parser.add_argument("--existing", type=str, default="output/potential_samples.pkl",
                        help="Existing samples file to compute bucket deficits from")
    parser.add_argument("--ratio-buckets", type=int, default=10,
                        help="Number of ratio buckets for deficit sampling")
    parser.add_argument("--parallel-targets", type=int, default=1,
                        help="Number of targets to run in parallel (each uses its own worker pool)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TopoFlow Potential Net — Trajectory Collection")
    print(f"  n_targets={args.n_targets}  pop={args.pop_size}  gens={args.generations}"
          + (f"  [deficit-buckets target={args.bucket_target}]" if args.deficit_buckets else ""))
    print(f"  workers={args.workers}  mode={args.mode}  sim_max_frames={args.sim_max_frames}")
    print(f"  monitor={args.monitor}")
    print(f"  horizons: short={args.short_horizon}  medium={args.medium_horizon}")
    print("=" * 60)

    if args.monitor:
        summary = _monitor_memory()
        if summary is None:
            print("  [monitor] psutil not installed — resource monitoring disabled")
        else:
            print(f"  [monitor] initial: {summary}")

    targets: List[Tuple[int, int]] = []
    if args.deficit_buckets:
        targets = _build_deficit_targets(args.existing, args.bucket_target, args.ratio_buckets)
        if not targets:
            print("No deficit targets to sample. Exiting.")
            return
    else:
        seen: Set[float] = set()
        while len(targets) < args.n_targets:
            p = random.randint(1, 999)
            q = random.randint(p + 1, 1000)
            ratio = p / q
            if ratio not in seen:
                seen.add(ratio)
                targets.append((p, q))

    print(f"Sampled {len(targets)} unique target ratios")

    all_samples: List[Tuple[Tuple[Edge, ...], Tuple[int, int], float, float]] = []
    t_start = time.perf_counter()

    collect_fn = partial(
        collect_for_target,
        pop_size=args.pop_size,
        generations=args.generations,
        solver_workers=args.workers,
        solver_threads=args.solver_threads,
        mode=args.mode,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        short_horizon=args.short_horizon,
        medium_horizon=args.medium_horizon,
        min_samples=args.min_samples,
        sim_max_frames=args.sim_max_frames,
    )

    temp_dir = os.path.join(os.path.dirname(args.output) or ".", "_potential_temp")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    temp_files: List[str] = []

    pbar = tqdm(targets, desc="Collecting trajectories", unit="target", dynamic_ncols=True)
    for idx, target_pq in enumerate(pbar):
        try:
            samples = collect_fn(target_pq)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  Warning: failed for target {target_pq}: {e}")
            continue
        gc.collect()
        postfix = f"total_samples={len(all_samples):,}"
        if args.monitor:
            summary = _monitor_memory()
            if summary is not None:
                postfix += f" | {summary}"
        pbar.set_postfix_str(postfix)

        if (idx + 1) % args.save_every == 0 and all_samples:
            temp_path = os.path.join(temp_dir, f"batch_{idx + 1:05d}.pkl")
            with open(temp_path, "wb") as f:
                pickle.dump(all_samples, f)
            temp_files.append(temp_path)
            print(f"  Saved {len(all_samples):,} samples to {temp_path}")
            all_samples.clear()

    if all_samples:
        temp_path = os.path.join(temp_dir, "batch_final.pkl")
        with open(temp_path, "wb") as f:
            pickle.dump(all_samples, f)
        temp_files.append(temp_path)
        all_samples.clear()

    print(f"\nMerging {len(temp_files)} batch files...")
    merged: List = []
    total_saved = 0
    for tf in temp_files:
        with open(tf, "rb") as f:
            batch = pickle.load(f)
            merged.extend(batch)
            total_saved += len(batch)
    shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t_start
    print(f"Collection complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Total labeled samples: {len(merged):,}")
    if merged:
        print(f"Avg samples per target: {len(merged) / max(len(targets), 1):.0f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved to: {args.output}")

    if merged:
        p_short_vals = [s[2] for s in merged]
        p_medium_vals = [s[3] for s in merged]
        print(f"Potential (short):  min={np.min(p_short_vals):.4f}  "
              f"mean={np.mean(p_short_vals):.4f}  max={np.max(p_short_vals):.4f}")
        print(f"Potential (medium): min={np.min(p_medium_vals):.4f}  "
              f"mean={np.mean(p_medium_vals):.4f}  max={np.max(p_medium_vals):.4f}")


if __name__ == "__main__":
    main()
