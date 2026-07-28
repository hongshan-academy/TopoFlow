import multiprocessing
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple

from graph import Graph
from config import DEFAULT_CONFIG as _cfg

from ga.fitness import make_evaluate
from ga.generation import generate_strict_graph
from ga.mutation import MUTATION_FNS
from ga.utils import edges_to_tuple, tuple_to_graph


# ==================== Parallel worker (per-process LRU cache) ====================

_worker_eval_fn = None


def _worker_init(v_target: float, threads: int):
    global _worker_eval_fn
    _worker_eval_fn = make_evaluate(v_target, threads)


def _worker_eval(edges_tuple: Tuple[Tuple[str, str], ...]) -> Tuple:
    global _worker_eval_fn
    try:
        t0 = time.perf_counter()
        err, nodes = _worker_eval_fn(edges_tuple)
        solve_time = time.perf_counter() - t0
        return (edges_tuple, err, nodes, solve_time)
    except Exception:
        return (edges_tuple, float("inf"), 0, 0.0)


# ==================== Individual helpers ====================

def _random_individual() -> Tuple[Tuple[str, str], ...]:
    n_internal = random.choices(
        _cfg.internal_nodes_choices,
        weights=_cfg.internal_nodes_weights,
        k=1,
    )[0]
    graph = generate_strict_graph(n_internal)
    assert graph.is_valid(strict=True), "Generated invalid graph"
    return edges_to_tuple(graph)


def _mutate(
    individual: Tuple[Tuple[str, str], ...],
    weights: Tuple[float, ...],
    max_tries: int = 5,
) -> Tuple[Tuple[str, str], ...]:
    for _ in range(max_tries):
        fn = random.choices(MUTATION_FNS, weights=weights, k=1)[0]
        result = fn(individual)
        if result is not None:
            return result
    return individual


# ==================== Selection ====================

def _tournament_select(
    population: List,
    k: int,
    tournsize: int = 2,
) -> List:
    if k <= 0:
        return []
    selected: List = []
    for _ in range(k):
        candidates = random.sample(population, min(tournsize, len(population)))
        champion = min(candidates, key=lambda ind: ind[1])
        selected.append(champion)
    return selected


# ==================== Batch evaluation with timing ====================

def _eval_batch(
    pool,
    indivs: List[Tuple[Tuple[str, str], ...]],
):
    if not indivs:
        return {}, [], 0.0, 0.0, 0.0

    t_start = time.perf_counter()
    it = pool.imap_unordered(_worker_eval, indivs)

    results: Dict = {}
    solver_times: List[float] = []
    first_arrival = None
    last_arrival = None

    for edges_tuple, err, nodes, solve_time in it:
        now = time.perf_counter()
        if first_arrival is None:
            first_arrival = now
        last_arrival = now
        results[edges_tuple] = (err, nodes)
        solver_times.append(solve_time)

    t_dispatch = (first_arrival - t_start) if first_arrival is not None else 0.0
    t_collect = (last_arrival - first_arrival) if (first_arrival is not None and last_arrival is not None) else 0.0
    t_total = (last_arrival - t_start) if last_arrival is not None else 0.0

    return results, solver_times, t_dispatch, t_collect, t_total


# ==================== Timing report ====================

def _report_timing(
    gen: int,
    timing: Dict,
    solver_times: List[float],
    n_mut: int,
    n_imm: int,
    n_eval: int,
    n_workers: int,
):
    t_total = timing.get("total", 0.0)
    t_sel = timing.get("select", 0.0)
    t_mut = timing.get("mutate", 0.0)
    t_imm = timing.get("immigrate", 0.0)
    t_eval = timing.get("eval_total", 0.0)
    t_dispatch = timing.get("dispatch", 0.0)
    t_collect = timing.get("collect", 0.0)

    if solver_times:
        slv_min = min(solver_times)
        slv_med = statistics.median(solver_times)
        slv_max = max(solver_times)
        slv_std = statistics.stdev(solver_times) if len(solver_times) > 1 else 0.0
        sum_slv = sum(solver_times)
        ideal_parallel = sum_slv / n_workers
        utilization = ideal_parallel / t_eval * 100 if t_eval > 0 else 0.0
    else:
        slv_min = slv_med = slv_max = slv_std = 0.0
        sum_slv = 0.0
        ideal_parallel = 0.0
        utilization = 0.0

    bar = "=" * 60
    print(f"\n{bar}")
    print(
        f"Gen {gen:5d}  "
        f"total={t_total:.1f}s  "
        f"sel={t_sel*1000:.0f}ms  "
        f"mut={t_mut:.1f}s (n={n_mut})  "
        f"imm={t_imm*1000:.0f}ms (n={n_imm})"
    )
    print(
        f"  eval={t_eval:.1f}s (n={n_eval}, {n_workers}w)  "
        f"dispatch={t_dispatch*1000:.0f}ms  "
        f"collect={t_collect:.1f}s"
    )
    print(
        f"  solver: min={slv_min:.2f}s  "
        f"med={slv_med:.2f}s  "
        f"max={slv_max:.2f}s  "
        f"\u03c3={slv_std:.2f}s  "
        f"\u03a3={sum_slv:.1f}s"
    )
    print(
        f"  ideal_parallel={ideal_parallel:.1f}s  "
        f"util\u2248{utilization:.0f}%  "
        f"waste={t_total - t_eval - t_mut - t_sel - t_imm:.1f}s"
    )


# ==================== Evolution loop ====================

def _evolve_v2(
    population: List,
    pool,
    ngen: int,
    mutation_rate: float,
    mutation_weights: Tuple[float, ...],
    v_target: float,
    threads: int,
    elitism_count: int,
    immigration_rate: float,
    profile_interval: int,
):
    pop_size = len(population)
    n_immigrants = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0
    select_count = pop_size - elitism_count - n_immigrants

    best_edges = None
    best_fitness = (float("inf"), float("inf"))

    for gen in range(ngen):
        t_gen_start = time.perf_counter()

        # ---- 1. sort & elite ----
        population.sort(key=lambda ind: ind[1])
        elites: List = population[:elitism_count]

        current_best = population[0]
        if current_best[1] < best_fitness:
            best_edges = current_best[0]
            best_fitness = current_best[1]

        # ---- 2. selection ----
        t0 = time.perf_counter()
        if select_count > 0:
            selected = _tournament_select(population, select_count, tournsize=2)
        else:
            selected = []
        t_sel = time.perf_counter() - t0

        # ---- 3. mutation (serial) ----
        t0 = time.perf_counter()
        offspring: List = []
        n_mutated = 0
        for edges_tuple_val, _ in selected:
            if random.random() < mutation_rate:
                new_indiv = _mutate(edges_tuple_val, mutation_weights)
                offspring.append(new_indiv)
                n_mutated += 1
            else:
                offspring.append(edges_tuple_val)
        t_mut = time.perf_counter() - t0

        # ---- 4. immigration ----
        t0 = time.perf_counter()
        immigrants: List = []
        for _ in range(n_immigrants):
            immigrants.append(_random_individual())
        t_imm = time.perf_counter() - t0

        # ---- 5. dedupe & evaluate ----
        to_eval = list(set(offspring + immigrants))
        eval_results, solver_times, t_dispatch, t_collect, t_eval = _eval_batch(pool, to_eval)

        # ---- 6. rebuild population ----
        new_pop: List = []
        for edges_tuple_val, fitness_val in elites:
            new_pop.append((edges_tuple_val, fitness_val))
        for indiv in offspring:
            fit = eval_results.get(indiv)
            if fit is not None:
                new_pop.append((indiv, fit))
        for indiv in immigrants:
            fit = eval_results.get(indiv)
            if fit is not None:
                new_pop.append((indiv, fit))

        population = new_pop

        # ---- 7. report ----
        t_total = time.perf_counter() - t_gen_start

        if gen % profile_interval == 0 or gen == ngen - 1:
            timing = {
                "total": t_total,
                "select": t_sel,
                "mutate": t_mut,
                "immigrate": t_imm,
                "eval_total": t_eval,
                "dispatch": t_dispatch,
                "collect": t_collect,
            }
            _report_timing(
                gen,
                timing,
                solver_times,
                n_mut=n_mutated,
                n_imm=n_immigrants,
                n_eval=len(to_eval),
                n_workers=pool._processes if hasattr(pool, "_processes") else 1,
            )

            errs = [ind[1][0] for ind in population]
            nodes_list = [ind[1][1] for ind in population]
            best_idx = min(range(len(errs)), key=lambda i: (errs[i], nodes_list[i]))
            print(
                f"  fit: best=({errs[best_idx]:.6f},{nodes_list[best_idx]})  "
                f"avg={sum(errs)/len(errs):.6f}  "
                f"n={min(nodes_list)}~{max(nodes_list)}"
            )

    return best_edges, best_fitness


# ==================== Entry point ====================

def run_v2(
    v_target: float = None,
    pop_size: int = None,
    generations: int = None,
    mutation_rate: float = None,
    tournament_size: int = None,
    mutation_weights: Tuple[float, ...] = None,
    elitism_count: int = None,
    immigration_rate: float = None,
    solver_workers: int = None,
    solver_threads: int = None,
    profile_interval: int = 10,
) -> Tuple[Graph, float]:
    v_target = v_target if v_target is not None else _cfg.target_f
    pop_size = pop_size or _cfg.pop_size
    generations = generations or _cfg.generations
    mutation_rate = mutation_rate if mutation_rate is not None else _cfg.mutation_rate
    tournament_size = tournament_size or _cfg.tournament_size
    mutation_weights = mutation_weights or (0.225, 0.225, 0.225, 0.225, 0.1)
    elitism_count = elitism_count if elitism_count is not None else _cfg.elitism_count
    immigration_rate = immigration_rate if immigration_rate is not None else _cfg.immigration_rate
    n_workers = solver_workers if solver_workers is not None else _cfg.solver_workers
    threads = solver_threads if solver_threads is not None else _cfg.solver_threads

    print(f"{'='*60}")
    print(f"  TopoFlow GA v2  |  Target v = {v_target:.10f}")
    print(f"  pop={pop_size}  gen={generations}  mut_rate={mutation_rate}")
    print(f"  tournament_size={tournament_size}  elitism={elitism_count}  immigration={immigration_rate:.2f}")
    print(f"  solver_workers={n_workers}  solver_threads={threads}")
    print(f"  profile_interval={profile_interval}")
    print(f"{'='*60}")

    pool = multiprocessing.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(v_target, threads),
    )

    try:
        print(f"\nGenerating initial population ({pop_size} individuals)...")
        init_indivs: List = []
        for _ in range(pop_size):
            init_indivs.append(_random_individual())

        print(f"Evaluating initial population ({n_workers} workers)...")
        t_eval_start = time.perf_counter()
        eval_results, solver_times, t_dispatch, t_collect, t_eval_total = _eval_batch(pool, init_indivs)
        t_eval = time.perf_counter() - t_eval_start

        population: List = []
        for indiv in init_indivs:
            fitness = eval_results.get(indiv)
            if fitness is not None:
                population.append((indiv, fitness))

        errs = [f[0] for _, f in population]
        print(
            f"  done in {t_eval:.1f}s  |  "
            f"best={min(errs):.6f}  avg={sum(errs)/len(errs):.6f}  "
            f"worst={max(errs):.6f}\n"
        )

        t_start = time.perf_counter()
        best_edges, best_fitness = _evolve_v2(
            population,
            pool,
            generations,
            mutation_rate,
            mutation_weights,
            v_target,
            threads,
            elitism_count,
            immigration_rate,
            profile_interval,
        )
        elapsed = time.perf_counter() - t_start
    finally:
        pool.close()
        pool.join()

    best_graph = tuple_to_graph(best_edges)
    best_err, best_nodes = best_fitness

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Best fitness:      err={best_err:.10f}  nodes={best_nodes}")
    print(f"  Target v:          {v_target:.10f}")
    print(f"  Best graph:        {len(best_graph.nodes)} nodes, {len(best_graph.edges)} edges")
    print(f"  Valid (strict):    {best_graph.is_valid(strict=True)}")
    print(f"  Source:            {best_graph.sources}")
    print(f"  Sink:              {best_graph.sinks}")

    deg_counts: Dict = {}
    for n in best_graph.nodes:
        if n not in best_graph.sources and n not in best_graph.sinks:
            d = tuple(best_graph.degrees[n])
            deg_counts[d] = deg_counts.get(d, 0) + 1
    print(f"  Internal degrees:  {deg_counts}")

    return best_graph, best_err
