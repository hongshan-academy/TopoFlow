import concurrent.futures as cf
import random
import time
from functools import partial
from typing import Dict, Tuple

from deap import base, creator, tools
from tqdm import tqdm

from graph import Graph
from config import DEFAULT_CONFIG as _cfg

from ga.fitness import make_evaluate, evaluate_cached
from ga.generation import generate_strict_graph
from ga.mutation import MUTATION_FNS
from ga.utils import edges_to_tuple, tuple_to_graph

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", tuple, fitness=creator.FitnessMin)

# ── Worker-pool globals (set by _worker_init in each process) ──
_worker_eval_fn = None


def _worker_init(v_target: float, threads: int):
    global _worker_eval_fn
    _worker_eval_fn = make_evaluate(v_target, threads)


def _eval_one(edges_tuple: Tuple[Tuple[str, str], ...]) -> Tuple[float, int]:
    global _worker_eval_fn
    try:
        if _worker_eval_fn is not None:
            return _worker_eval_fn(edges_tuple)
        return evaluate_cached(edges_tuple, 0.0, 1)
    except Exception:
        return (float("inf"), 0)


# ── Individual helpers ──

def _generate_random_individual() -> creator.Individual:
    n_internal = random.choices(
        _cfg.internal_nodes_choices,
        weights=_cfg.internal_nodes_weights,
        k=1,
    )[0]
    graph = generate_strict_graph(n_internal)
    assert graph.is_valid(strict=True), "Generated invalid graph"
    return creator.Individual(edges_to_tuple(graph))


def _mutate_wrapper(
    individual: creator.Individual,
    weights: Tuple[float, ...],
    max_tries: int = 5,
) -> Tuple[creator.Individual]:
    for _ in range(max_tries):
        fn = random.choices(MUTATION_FNS, weights=weights, k=1)[0]
        result = fn(individual)
        if result is not None:
            return (creator.Individual(result),)
    return (individual,)


# ── Evolution with async evaluation ──

def _evolve(
    population: list,
    toolbox: base.Toolbox,
    ngen: int,
    mutation_rate: float,
    v_target: float,
    threads: int,
    executor: cf.ProcessPoolExecutor,
    elitism_count: int = 0,
    immigration_rate: float = 0.0,
    eval_timeout: float = 10.0,
) -> creator.Individual:
    hof = tools.HallOfFame(1)
    pop_size = len(population)

    _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0

    pending: Dict[cf.Future, int] = {}

    pbar = tqdm(total=ngen, desc="Evolving", unit="gen", dynamic_ncols=True)

    for gen in range(ngen):
        t0 = time.perf_counter()

        # ── 1. harvest already-completed futures (non-blocking) ──
        n_harvested = 0
        for fut in [f for f in pending if f.done()]:
            idx = pending.pop(fut)
            population[idx].fitness.values = fut.result()
            n_harvested += 1

        # ── 2. wait for remaining pending with timeout ──
        if pending:
            if eval_timeout is None:
                done, _ = cf.wait(list(pending))
            else:
                remaining = eval_timeout - (time.perf_counter() - t0)
                if remaining > 0:
                    done, _ = cf.wait(list(pending), timeout=remaining)
                else:
                    done = set()
            for fut in done:
                idx = pending.pop(fut)
                population[idx].fitness.values = fut.result()
                n_harvested += 1

        # ── 3. split ready / pending ──
        pending_idxs: set = set(pending.values())
        ready = [ind for i, ind in enumerate(population) if i not in pending_idxs]
        pending_count = len(pending_idxs)

        # ── 4. safety: if still zero ready, force-wait for at least one ──
        if not ready and pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                population[idx].fitness.values = fut.result()
                n_harvested += 1
            pending_idxs = set(pending.values())
            ready = [ind for i, ind in enumerate(population) if i not in pending_idxs]
            pending_count = len(pending_idxs)

        # ── 5. sort ready & elites ──
        ready.sort(key=lambda ind: ind.fitness.values)
        elites = [toolbox.clone(ind) for ind in ready[:elitism_count]]

        # ── 6. select from ready (fill remaining slots) ──
        n_immigrants = _immig_rate
        select_k = pop_size - elitism_count - pending_count - n_immigrants
        if select_k < 0:
            n_immigrants = max(0, pop_size - elitism_count - pending_count)
            select_k = 0

        selected: list = []
        if select_k > 0:
            selected = toolbox.select(ready, select_k)
        selected = [toolbox.clone(ind) for ind in selected]

        # ── 6. mutate ──
        n_mutated = 0
        for i, ind in enumerate(selected):
            if random.random() < mutation_rate:
                selected[i], = toolbox.mutate(ind)
                del selected[i].fitness.values
                n_mutated += 1

        # ── 7. immigrants ──
        immigrants = [toolbox.individual() for _ in range(n_immigrants)]

        # ── 8. rebuild population ──
        sorted_pending_idxs = sorted(pending_idxs)
        pending_individuals = [population[i] for i in sorted_pending_idxs]

        old_to_new: Dict[int, int] = {}
        for new_i, old_i in enumerate(sorted_pending_idxs):
            old_to_new[old_i] = elitism_count + new_i

        remapped: Dict[cf.Future, int] = {}
        for fut, old_idx in pending.items():
            remapped[fut] = old_to_new[old_idx]
        pending = remapped

        new_pop = elites + pending_individuals + selected + immigrants
        population[:] = new_pop

        hof.update(ready)

        # ── 9. submit new evaluations ──
        offset = elitism_count + len(pending_individuals)
        for i, ind in enumerate(selected + immigrants):
            fut = executor.submit(_eval_one, tuple(ind))
            pending[fut] = offset + i

        # ── 10. stats ──
        errs_list = [ind.fitness.values[0] for ind in ready]
        nodes_list = [ind.fitness.values[1] for ind in ready]
        if errs_list:
            best_idx = min(range(len(errs_list)), key=lambda i: (errs_list[i], nodes_list[i]))
            best_err = errs_list[best_idx]
            best_n = nodes_list[best_idx]
            avg_err = sum(errs_list) / len(errs_list)
        else:
            best_err = float("inf")
            best_n = 0
            avg_err = float("inf")

        elapsed = time.perf_counter() - t0

        pbar.set_postfix_str(
            f"best=({best_err:.6f},{best_n}) "
            f"avg={avg_err:.6f} "
            f"n={min(nodes_list) if nodes_list else 0}~{max(nodes_list) if nodes_list else 0} "
            f"mut={n_mutated}/{len(selected)} "
            f"imm={n_immigrants} "
            f"R={len(ready)} P={pending_count} H={n_harvested} "
            f"t={elapsed:.1f}s"
        )
        pbar.update(1)

    pbar.close()
    return hof[0]


# ── Entry point ──

def run(
    v_target: float = None,
    pop_size: int = None,
    generations: int = None,
    mutation_rate: float = None,
    tournament_size: int = None,
    mutation_weights: Tuple[float, ...] = None,
    elitism_count: int = None,
    immigration_rate: float = None,
    eval_timeout: float = None,
    solver_workers: int = None,
    solver_threads: int = None,
) -> Tuple[Graph, float]:
    v_target = v_target if v_target is not None else _cfg.target_f
    pop_size = pop_size or _cfg.pop_size
    generations = generations or _cfg.generations
    mutation_rate = mutation_rate if mutation_rate is not None else _cfg.mutation_rate
    tournament_size = tournament_size or _cfg.tournament_size
    mutation_weights = mutation_weights or (0.225, 0.225, 0.225, 0.225, 0.1)
    elitism_count = elitism_count if elitism_count is not None else _cfg.elitism_count
    immigration_rate = immigration_rate if immigration_rate is not None else _cfg.immigration_rate
    eval_timeout = eval_timeout if eval_timeout is not None else _cfg.eval_timeout
    n_workers = solver_workers if solver_workers is not None else _cfg.solver_workers
    threads = solver_threads if solver_threads is not None else _cfg.solver_threads

    print(f"{'='*60}")
    print(f"  TopoFlow GA (async) |  Target v = {v_target:.10f}")
    print(f"  pop={pop_size}  gen={generations}  mut_rate={mutation_rate}")
    print(f"  tournament_size={tournament_size}  elitism={elitism_count}  immigration={immigration_rate:.2f}")
    print(f"  eval_timeout={eval_timeout}s  workers={n_workers}  threads={threads}")
    print(f"{'='*60}")

    toolbox = base.Toolbox()
    toolbox.register("individual", _generate_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mutate", partial(_mutate_wrapper, weights=mutation_weights))

    print(f"\nGenerating initial population ({pop_size} individuals)...")
    population = toolbox.population(n=pop_size)

    print(f"Evaluating initial population ({n_workers} workers)...")
    t_eval = time.perf_counter()

    with cf.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(v_target, threads),
    ) as executor:
        fut_to_idx = {
            executor.submit(_eval_one, tuple(ind)): i
            for i, ind in enumerate(population)
        }
        for fut in tqdm(
            cf.as_completed(fut_to_idx),
            total=pop_size,
            desc="  Evaluating",
            unit="ind",
            dynamic_ncols=True,
        ):
            idx = fut_to_idx[fut]
            population[idx].fitness.values = fut.result()

        t_eval = time.perf_counter() - t_eval

        initial_fits = [ind.fitness.values[0] for ind in population]
        print(
            f"  done in {t_eval:.1f}s  |  "
            f"best={min(initial_fits):.6f}  avg={sum(initial_fits)/len(initial_fits):.6f}  "
            f"worst={max(initial_fits):.6f}\n"
        )

        t_start = time.perf_counter()
        best_ind = _evolve(
            population, toolbox, generations, mutation_rate,
            v_target, threads, executor, elitism_count, immigration_rate, eval_timeout,
        )
        elapsed = time.perf_counter() - t_start

    best_graph = tuple_to_graph(best_ind)
    best_err = best_ind.fitness.values[0]
    best_nodes = best_ind.fitness.values[1]

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

    deg_counts = {}
    for n in best_graph.nodes:
        if n not in best_graph.sources and n not in best_graph.sinks:
            d = tuple(best_graph.degrees[n])
            deg_counts[d] = deg_counts.get(d, 0) + 1
    print(f"  Internal degrees:  {deg_counts}")

    return best_graph, best_err
