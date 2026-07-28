import concurrent.futures as cf
import random
import time
from functools import partial
from typing import Dict, Optional, Tuple

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


def _worker_init(target_pq: Tuple[int, int], threads: int, max_denominator: int = 10000):
    global _worker_eval_fn
    _worker_eval_fn = make_evaluate(target_pq, threads, max_denominator=max_denominator)


def _eval_one(edges_tuple: Tuple[Tuple[str, str], ...]) -> Tuple[float, int]:
    global _worker_eval_fn
    try:
        if _worker_eval_fn is not None:
            return _worker_eval_fn(edges_tuple)
        return evaluate_cached(edges_tuple, (0, 1), 1)
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
    threads: int,
    executor: cf.ProcessPoolExecutor,
    elitism_count: int = 0,
    immigration_rate: float = 0.0,
    eval_timeout: Optional[float] = None,
) -> creator.Individual:
    hof = tools.HallOfFame(5)
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

        # ── 2. dynamic wait: harvest until enough ready ──
        if pending:
            t_wait = time.perf_counter()
            while pending:
                pending_idxs = set(pending.values())
                ready_check = [ind for i, ind in enumerate(population) if i not in pending_idxs]
                desired_ready = max(elitism_count + 1, pop_size - executor._max_workers)
                if len(ready_check) >= desired_ready:
                    break
                if eval_timeout is not None:
                    remaining = eval_timeout - (time.perf_counter() - t_wait)
                    if remaining <= 0:
                        break
                    done, _ = cf.wait(list(pending), timeout=remaining, return_when=cf.FIRST_COMPLETED)
                else:
                    done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
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
            f"best=({best_err:.0f},{best_n:.0f}) "
            f"avg={avg_err:.6f} "
            f"n={min(nodes_list) if nodes_list else 0}~{max(nodes_list) if nodes_list else 0} "
            f"mut={n_mutated}/{len(selected)} "
            f"imm={n_immigrants} "
            f"R={len(ready)} P={pending_count} H={n_harvested} "
            f"t={elapsed:.1f}s"
        )
        pbar.update(1)

    pbar.close()
    return hof


# ── Entry point ──

def run(
    target_pq: Tuple[int, int] = None,
    pop_size: int = None,
    generations: int = None,
    mutation_rate: float = None,
    tournament_size: int = None,
    mutation_weights: Tuple[float, ...] = None,
    elitism_count: int = None,
    immigration_rate: float = None,
    eval_timeout: Optional[float] = None,
    solver_workers: int = None,
    solver_threads: int = None,
    max_denominator: int = 10000,
    output_path: Optional[str] = None,
) -> list[dict]:
    target_pq = target_pq if target_pq is not None else _cfg.target_pq
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
    output_path = output_path or _cfg.output_path

    print(f"{'='*60}")
    print(f"  TopoFlow GA (async) |  Target = {target_pq[0]}/{target_pq[1]}")
    print(f"  pop={pop_size}  gen={generations}  mut_rate={mutation_rate}")
    print(f"  tournament_size={tournament_size}  elitism={elitism_count}  immigration={immigration_rate:.2f}")
    print(f"  eval_timeout={eval_timeout if eval_timeout is not None else 'dynamic'}s  workers={n_workers}  threads={threads}")
    print(f"  max_denominator={max_denominator}")
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
        initargs=(target_pq, threads, max_denominator),
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
        hof = _evolve(
            population, toolbox, generations, mutation_rate,
            threads, executor, elitism_count, immigration_rate, eval_timeout,
        )
        elapsed = time.perf_counter() - t_start

    results = []
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/60:.1f}min)")

    for rank, ind in enumerate(hof):
        graph = tuple_to_graph(ind)
        err = ind.fitness.values[0]
        nodes = ind.fitness.values[1]

        result = {
            "rank": rank + 1,
            "error": int(err),
            "nodes": int(nodes),
            "target": {"p": target_pq[0], "q": target_pq[1]},
            "graph": {
                "nodes": sorted(graph.nodes),
                "edges": [list(e) for e in graph.edges],
            },
        }
        results.append(result)

        print(f"  #{rank + 1}:  err={err:.0f}  nodes={nodes}  |  "
              f"{len(graph.nodes)} nodes, {len(graph.edges)} edges  |  "
              f"valid(strict)={graph.is_valid(strict=True)}")

    import json
    import os
    out_path = output_path or os.path.join("output", "ga_top5.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")

    return results
