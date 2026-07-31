import concurrent.futures as cf
import random
import time

import numpy as np
from functools import partial
from typing import Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rf_surrogate.model import SurrogateRF
    from rf_surrogate.archive import SurrogateArchive

from deap import base, creator, tools
from tqdm import tqdm

from config import DEFAULT_CONFIG as _cfg

from ga.chromosome import make_random_chromosome, decode
from ga.crossover import crossover_pmx
from ga.fitness import make_evaluate, evaluate_cached
from ga.history import History
from ga.mutation import MUTATION_FNS

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", tuple, fitness=creator.FitnessMin)

_worker_eval_fn = None


def _worker_init(target_pq: Tuple[int, int], threads: int, max_denominator: int = 10000, mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP') -> None:
    global _worker_eval_fn
    _worker_eval_fn = make_evaluate(target_pq, threads, max_denominator=max_denominator, mode=mode)


def _eval_one(chromosome: Tuple[int, ...]) -> Tuple[float, int]:
    global _worker_eval_fn
    try:
        if _worker_eval_fn is not None:
            return _worker_eval_fn(chromosome)  # type: ignore[no-any-return]
        return evaluate_cached(chromosome, (0, 1), 1)
    except Exception:
        return (float("inf"), 0)


def _generate_random_individual() -> creator.Individual:
    for _ in range(_cfg.individual_retries):
        n_internal = random.choices(
            _cfg.internal_nodes_choices,
            weights=_cfg.internal_nodes_weights,
            k=1,
        )[0]
        chrom = make_random_chromosome(n_internal)
        g = decode(chrom)
        if g.is_valid(strict=True) and len(g.edges) <= _cfg.max_edges:
            return creator.Individual(chrom)
    n_internal = min(_cfg.internal_nodes_choices)
    chrom = make_random_chromosome(n_internal)
    return creator.Individual(chrom)


def _mutate_wrapper(
    individual: creator.Individual,
    weights: Tuple[float, ...],
    max_tries: int = 5,
) -> Tuple[creator.Individual]:
    for _ in range(max_tries):
        fn = random.choices(MUTATION_FNS, weights=weights, k=1)[0]
        result = fn(individual)
        g = decode(result)
        if g.is_valid(strict=True) and len(g.edges) <= _cfg.max_edges:
            return (creator.Individual(result),)
    return (individual,)


def _evolve(
    population: List[creator.Individual],
    toolbox: base.Toolbox,
    ngen: int,
    mutation_rate: float,
    crossover_rate: float,
    threads: int,
    executor: cf.ProcessPoolExecutor,
    elitism_count: int = 0,
    immigration_rate: float = 0.0,
    eval_timeout: Optional[float] = None,
    history: Optional[History] = None,
    surrogate: Optional["SurrogateRF"] = None,
    archive: Optional["SurrogateArchive"] = None,
    surrogate_top_fraction: float = 0.25,
    surrogate_random_eval_fraction: float = 0.05,
    surrogate_retrain_interval: int = 5,
    surrogate_warmup_samples: int = 80,
) -> tools.HallOfFame:
    hof = tools.HallOfFame(5)
    pop_size = len(population)
    n_workers = executor._max_workers if hasattr(executor, '_max_workers') else 1

    _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0

    pending: Dict[cf.Future[Tuple[float, int]], int] = {}
    stagnation = 0
    best_ever_err = float("inf")

    pbar = tqdm(total=ngen, desc="Evolving", unit="gen", dynamic_ncols=True)

    for gen in range(ngen):
        if gen % surrogate_retrain_interval == 0 and surrogate is not None and archive is not None and archive.size() >= surrogate_warmup_samples:
            X, y = archive.get_data()
            surrogate.fit(X, y)

        t0 = time.perf_counter()

        def _harvest_one(fut: cf.Future[Tuple[float, int]]) -> Optional[int]:
            idx = pending.pop(fut)
            fitness = fut.result()
            if fitness[0] == float("inf") and random.random() >= _cfg.infeasible_throughput:
                new_ind = toolbox.individual()
                population[idx] = new_ind
                new_fut = executor.submit(_eval_one, tuple(new_ind))
                pending[new_fut] = idx
                return None
            population[idx].fitness.values = fitness
            return idx

        def _surrogate_filter(individuals, top_frac, rand_frac):
            from rf_surrogate.features import extract_features
            if not surrogate.is_ready():
                return set(range(len(individuals)))
            n = len(individuals)
            if n == 0:
                return set()
            features_list = [extract_features(tuple(ind)) for ind in individuals]
            features = np.array(features_list, dtype=np.float64)
            predictions = surrogate.predict(features)
            sorted_idx = np.argsort(predictions)
            n_top = max(1, int(n * top_frac))
            eval_set = set(sorted_idx[:n_top].tolist())
            if rand_frac > 0:
                remaining = [i for i in range(n) if i not in eval_set]
                n_rand = min(int(n * rand_frac), len(remaining))
                if n_rand > 0:
                    chosen = np.random.choice(remaining, n_rand, replace=False).tolist()
                    eval_set.update(chosen)
            for i, ind in enumerate(individuals):
                g = decode(tuple(ind))
                pred_err = max(0.0, float(predictions[i]))
                ind.fitness.values = (pred_err, len(g.edges))
            return eval_set

        n_harvested = 0
        for fut in [f for f in pending if f.done()]:
            idx = _harvest_one(fut)
            if idx is not None:
                fit = population[idx].fitness.values[0]
                if archive is not None and fit != float("inf"):
                    archive.add(tuple(population[idx]), fit)
                n_harvested += 1

        if pending:
            t_wait = time.perf_counter()
            while pending:
                pending_idxs_set: Set[int] = set(pending.values())
                ready_check = [ind for i, ind in enumerate(population) if i not in pending_idxs_set]
                desired_ready = max(elitism_count + 1, pop_size - n_workers)
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
                    idx = _harvest_one(fut)
                    if idx is not None:
                        fit = population[idx].fitness.values[0]
                        if archive is not None and fit != float("inf"):
                            archive.add(tuple(population[idx]), fit)
                        n_harvested += 1

        new_pending_idxs: Set[int] = set(pending.values())
        ready = [ind for i, ind in enumerate(population) if i not in new_pending_idxs]
        pending_count = len(new_pending_idxs)

        if not ready and pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx = _harvest_one(fut)
                if idx is not None:
                    fit = population[idx].fitness.values[0]
                    if archive is not None and fit != float("inf"):
                        archive.add(tuple(population[idx]), fit)
                    n_harvested += 1
            new_pending_idxs = set(pending.values())
            ready = [ind for i, ind in enumerate(population) if i not in new_pending_idxs]
            pending_count = len(new_pending_idxs)

        ready.sort(key=lambda ind: ind.fitness.values)
        elites = [toolbox.clone(ind) for ind in ready[:elitism_count]]

        n_immigrants = _immig_rate
        select_k = pop_size - elitism_count - pending_count - n_immigrants
        if select_k < 0:
            n_immigrants = max(0, pop_size - elitism_count - pending_count)
            select_k = 0

        selected: List[creator.Individual] = []
        if select_k > 0:
            selected = toolbox.select(ready, select_k)
        selected = [toolbox.clone(ind) for ind in selected]

        n_crossed = 0
        for i in range(0, len(selected) - 1, 2):
            if random.random() < crossover_rate:
                c1, c2 = toolbox.mate(selected[i], selected[i + 1])
                selected[i] = creator.Individual(c1)
                selected[i + 1] = creator.Individual(c2)
                del selected[i].fitness.values
                del selected[i + 1].fitness.values
                n_crossed += 2

        n_mutated = 0
        for i, ind in enumerate(selected):
            if random.random() < mutation_rate:
                selected[i], = toolbox.mutate(ind)
                del selected[i].fitness.values
                n_mutated += 1

        immigrants = [toolbox.individual() for _ in range(n_immigrants)]

        sorted_pending_idxs = sorted(new_pending_idxs)
        pending_individuals = [population[i] for i in sorted_pending_idxs]

        old_to_new: Dict[int, int] = {}
        for new_i, old_i in enumerate(sorted_pending_idxs):
            old_to_new[old_i] = elitism_count + new_i

        remapped: Dict[cf.Future[Tuple[float, int]], int] = {}
        for fut, old_idx in pending.items():
            remapped[fut] = old_to_new[old_idx]
        pending = remapped

        new_pop = elites + pending_individuals + selected + immigrants
        population[:] = new_pop

        hof.update(ready)

        new_individuals = selected + immigrants

        eval_set = set(range(len(new_individuals)))
        if surrogate is not None and surrogate.is_ready():
            eval_set = _surrogate_filter(new_individuals, surrogate_top_fraction, surrogate_random_eval_fraction)

        offset = elitism_count + len(pending_individuals)
        for i, ind in enumerate(new_individuals):
            if i in eval_set:
                fut = executor.submit(_eval_one, tuple(ind))
                pending[fut] = offset + i

        errs_list = [ind.fitness.values[0] for ind in ready]
        nodes_list = [ind.fitness.values[1] for ind in ready]

        edges_list = []
        for ind in ready:
            g = decode(ind)
            edges_list.append(len(g.edges))

        if errs_list:
            best_idx = min(range(len(errs_list)), key=lambda i: (errs_list[i], nodes_list[i]))
            best_err = errs_list[best_idx]
            best_n = nodes_list[best_idx]
            best_ind = ready[best_idx]
            avg_err = sum(errs_list) / len(errs_list)
        else:
            best_err = float("inf")
            best_n = 0
            best_ind = None
            avg_err = float("inf")

        elapsed = time.perf_counter() - t0

        if history is not None:
            min_n = min(nodes_list) if nodes_list else 0
            max_n = max(nodes_list) if nodes_list else 0
            max_e = max(edges_list) if edges_list else 0

            best_graph_edges: Optional[List[List[str]]] = None
            if best_ind is not None:
                graph = decode(best_ind)
                best_graph_edges = [list(e) for e in graph.edges]
                history.record_best(
                    gen=gen,
                    edges=best_graph_edges,
                    fitness=(float(best_err), int(best_n)),
                    graph_nodes=len(graph.nodes),
                    graph_edges=len(graph.edges),
                )

            history.record_gen(
                gen=gen,
                best_error=best_err,
                best_nodes=best_n,
                avg_error=avg_err,
                min_nodes=min_n,
                max_nodes=max_n,
                max_edges=max_e,
                n_crossed=n_crossed,
                n_mutated=n_mutated,
                n_selected=len(selected),
                n_immigrants=n_immigrants,
                n_ready=len(ready),
                n_pending=pending_count,
                n_harvested=n_harvested,
                elapsed_sec=elapsed,
                best_edges=best_graph_edges,
            )

        if best_err < best_ever_err - 1e-9:
            best_ever_err = best_err
            stagnation = 0
            _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0
        else:
            stagnation += 1

        if 0 < stagnation <= _cfg.stagnation_restart and stagnation % _cfg.stagnation_interval == 0:
            _immig_rate = max(1, int(pop_size * _cfg.stagnation_boost_ratio))

        if stagnation >= _cfg.stagnation_restart:
            ready_sorted = sorted(ready, key=lambda ind: ind.fitness.values)
            n_keep = min(_cfg.restart_survivors, len(ready_sorted))
            survivors = [toolbox.clone(ind) for ind in ready_sorted[:n_keep]]
            new_randoms = [toolbox.individual() for _ in range(pop_size - n_keep)]
            new_indiv_list = [creator.Individual(c) for c in new_randoms]
            eval_set = set(range(len(new_indiv_list)))
            if surrogate is not None and surrogate.is_ready():
                eval_set = _surrogate_filter(new_indiv_list, surrogate_top_fraction, surrogate_random_eval_fraction)
            population[:] = survivors + new_indiv_list
            pending.clear()
            for i, ind in enumerate(new_indiv_list):
                if i in eval_set:
                    fut = executor.submit(_eval_one, tuple(ind))
                    pending[fut] = n_keep + i
            best_ever_err = float("inf")
            stagnation = 0
            _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0
            pbar.update(1)
            continue

        pbar.set_postfix_str(
            f"best=({best_err:.6f}, {best_n:.0f}) "
            f"avg={avg_err:.6f} "
            f"n={min(nodes_list) if nodes_list else 0}~{max(nodes_list) if nodes_list else 0} "
            f"x={n_crossed} mut={n_mutated}/{len(selected)} "
            f"imm={n_immigrants} "
            f"R={len(ready)} P={pending_count} H={n_harvested} "
            f"t={elapsed:.1f}s"
        )
        pbar.update(1)

    pbar.close()
    return hof


def run(
    target_pq: Optional[Tuple[int, int]] = None,
    pop_size: Optional[int] = None,
    generations: Optional[int] = None,
    mutation_rate: Optional[float] = None,
    crossover_rate: Optional[float] = None,
    tournament_size: Optional[int] = None,
    mutation_weights: Optional[Tuple[float, ...]] = None,
    elitism_count: Optional[int] = None,
    immigration_rate: Optional[float] = None,
    eval_timeout: Optional[float] = None,
    solver_workers: Optional[int] = None,
    solver_threads: Optional[int] = None,
    max_denominator: int = 10000,
    mode: Optional[str] = None,
    output_path: Optional[str] = None,
    surrogate_enabled: Optional[bool] = None,
    surrogate_top_fraction: Optional[float] = None,
    surrogate_random_eval_fraction: Optional[float] = None,
    surrogate_warmup_samples: Optional[int] = None,
    surrogate_retrain_interval: Optional[int] = None,
    surrogate_n_estimators: Optional[int] = None,
    surrogate_max_depth: Optional[int] = None,
) -> List[Dict[str, object]]:
    target_pq = target_pq if target_pq is not None else _cfg.target_pq
    pop_size = pop_size or _cfg.pop_size
    generations = generations or _cfg.generations
    mutation_rate = mutation_rate if mutation_rate is not None else _cfg.mutation_rate
    crossover_rate = crossover_rate if crossover_rate is not None else _cfg.crossover_rate
    tournament_size = tournament_size or _cfg.tournament_size
    mutation_weights = mutation_weights or tuple(_cfg.mutation_weights)
    elitism_count = elitism_count if elitism_count is not None else _cfg.elitism_count
    immigration_rate = immigration_rate if immigration_rate is not None else _cfg.immigration_rate
    eval_timeout = eval_timeout if eval_timeout is not None else _cfg.eval_timeout
    n_workers = solver_workers if solver_workers is not None else _cfg.solver_workers
    threads = solver_threads if solver_threads is not None else _cfg.solver_threads
    mode = mode if mode is not None else _cfg.mode
    output_path = output_path or _cfg.output_path
    surrogate_enabled = surrogate_enabled if surrogate_enabled is not None else _cfg.surrogate_enabled
    surrogate_top_fraction = surrogate_top_fraction if surrogate_top_fraction is not None else _cfg.surrogate_top_fraction
    surrogate_random_eval_fraction = surrogate_random_eval_fraction if surrogate_random_eval_fraction is not None else _cfg.surrogate_random_eval_fraction
    surrogate_warmup_samples = surrogate_warmup_samples if surrogate_warmup_samples is not None else _cfg.surrogate_warmup_samples
    surrogate_retrain_interval = surrogate_retrain_interval if surrogate_retrain_interval is not None else _cfg.surrogate_retrain_interval
    surrogate_n_estimators = surrogate_n_estimators if surrogate_n_estimators is not None else _cfg.surrogate_n_estimators
    surrogate_max_depth = surrogate_max_depth if surrogate_max_depth is not None else _cfg.surrogate_max_depth

    print(f"{'='*60}")
    print(f"  TopoFlow GA |  Target = {target_pq[0]}/{target_pq[1]}")
    print(f"  mode={mode}  pop={pop_size}  gen={generations}  mut_rate={mutation_rate}  x_rate={crossover_rate}")
    print(f"  tournament={tournament_size}  elitism={elitism_count}  immigration={immigration_rate:.2f}")
    print(f"  workers={n_workers}  threads={threads}")
    print(f"{'='*60}")

    history = History()
    history.target_pq = target_pq
    history.set_params(
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        mutation_weights=list(mutation_weights),
        elitism_count=elitism_count,
        immigration_rate=immigration_rate,
        eval_timeout=eval_timeout,
        solver_workers=n_workers,
        solver_threads=threads,
        max_denominator=max_denominator,
        mode=mode,
    )

    toolbox = base.Toolbox()
    toolbox.register("individual", _generate_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", crossover_pmx)
    toolbox.register("mutate", partial(_mutate_wrapper, weights=mutation_weights, max_tries=_cfg.mutation_max_tries))

    print(f"\nGenerating initial population ({pop_size} individuals)...")
    population = toolbox.population(n=pop_size)

    print(f"Evaluating initial population ({n_workers} workers)...")
    t_eval = time.perf_counter()

    with cf.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,  # type: ignore[arg-type]
        initargs=(target_pq, threads, max_denominator, mode),  # type: ignore[arg-type]
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

        replace_idxs = [
            i for i, ind in enumerate(population)
            if ind.fitness.values[0] == float("inf") and random.random() >= _cfg.infeasible_throughput
        ]
        if replace_idxs:
            for i in replace_idxs:
                population[i] = toolbox.individual()
            fut_to_idx = {
                executor.submit(_eval_one, tuple(population[i])): i
                for i in replace_idxs
            }
            for _ in tqdm(
                cf.as_completed(fut_to_idx),
                total=len(replace_idxs),
                desc="  Re-eval",
                unit="ind",
                dynamic_ncols=True,
            ):
                pass
            for fut, idx in fut_to_idx.items():
                population[idx].fitness.values = fut.result()

        t_eval = time.perf_counter() - t_eval

        initial_fits = [ind.fitness.values[0] for ind in population]
        print(
            f"  done in {t_eval:.1f}s  |  "
            f"best={min(initial_fits):.6f}  avg={sum(initial_fits)/len(initial_fits):.6f}  "
            f"worst={max(initial_fits):.6f}\n"
        )

        archive = None
        surrogate = None
        if surrogate_enabled:
            from rf_surrogate.archive import SurrogateArchive
            from rf_surrogate.model import SurrogateRF
            archive = SurrogateArchive()
            for ind in population:
                fit = ind.fitness.values[0]
                if fit != float("inf"):
                    archive.add(tuple(ind), fit)
            if archive.size() >= surrogate_warmup_samples:
                surrogate = SurrogateRF(
                    n_estimators=surrogate_n_estimators,
                    max_depth=surrogate_max_depth,
                )
                X, y = archive.get_data()
                surrogate.fit(X, y)

        t_start = time.perf_counter()
        hof = _evolve(
            population, toolbox, generations, mutation_rate, crossover_rate,
            threads, executor, elitism_count, immigration_rate, eval_timeout,
            history=history,
            surrogate=surrogate,
            archive=archive,
            surrogate_top_fraction=surrogate_top_fraction,
            surrogate_random_eval_fraction=surrogate_random_eval_fraction,
            surrogate_retrain_interval=surrogate_retrain_interval,
            surrogate_warmup_samples=surrogate_warmup_samples,
        )
        elapsed = time.perf_counter() - t_start

    results = []
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/60:.1f}min)")

    for rank, ind in enumerate(hof):
        graph = decode(ind)
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

        print(f"  #{rank + 1}:  err={err:.6f}  nodes={nodes}  |  "
              f"{len(graph.nodes)} nodes, {len(graph.edges)} edges  |  "
              f"valid(strict)={graph.is_valid(strict=True)}")

    import json
    import os
    out_path = output_path or os.path.join("output", "ga_top5.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")

    if surrogate is not None and surrogate.is_ready() and archive is not None:
        import pickle
        os.makedirs("output", exist_ok=True)
        X, y = archive.get_data()
        import numpy as np
        np.save("output/rf_archive_X.npy", X)
        np.save("output/rf_archive_y.npy", y)
        with open("output/rf_model.pkl", "wb") as f:
            pickle.dump(surrogate._model, f)
        print(f"  RF model & archive saved to output/  (n_samples={archive.size()})")

    history.to_json(_cfg.history_path)
    print(f"\n{history.summary()}")

    return results
