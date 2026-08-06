import concurrent.futures as cf
import os
import random
import time

import numpy as np
from functools import partial
from typing import Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rf_surrogate.model import SurrogateRF
    from nn_surrogate.model import SurrogateGNN
    from rf_surrogate.archive import SurrogateArchive
    from potential_net.model import PotentialNet

from deap import base, creator, tools
from tqdm import tqdm

from config import DEFAULT_CONFIG as _cfg

from graph import Graph, Edge
from ga.generation import generate_strict_graph
from ga.crossover import CROSSOVER_FNS
from ga.fitness import make_evaluate, evaluate_cached
from ga.history import History
from ga.mutation import MUTATION_FNS

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", tuple, fitness=creator.FitnessMin)

_worker_eval_fn = None


def _worker_init(target_pq: Tuple[int, int], threads: int, max_denominator: int = 10000, mode: Literal['MILP', 'simulation', 'mixed'] = 'MILP') -> None:
    global _worker_eval_fn
    _worker_eval_fn = make_evaluate(target_pq, threads, max_denominator=max_denominator, mode=mode)


def _eval_one(edges_tuple: Tuple[Edge, ...]) -> Tuple[float, int]:
    global _worker_eval_fn
    try:
        if _worker_eval_fn is not None:
            result = _worker_eval_fn(edges_tuple)
            return (result[0], result[1])  # strip flow_ratio
        return evaluate_cached(edges_tuple, (0, 1), 1)[:2]
    except Exception:
        return (float("inf"), 0)


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
    surrogate = None,
    archive: Optional["SurrogateArchive"] = None,
    surrogate_top_fraction: float = 0.25,
    surrogate_random_eval_fraction: float = 0.05,
    surrogate_retrain_interval: int = 5,
    surrogate_warmup_samples: int = 80,
    surrogate_type: str = "rf",
    target_pq: Tuple[int, int] = (325, 799),
    pretrained_loaded: bool = False,
    potential_model: Optional["PotentialNet"] = None,
    mutation_budget_base: int = 1,
    mutation_budget_max_extra: int = 4,
    budget_short_weight: float = 0.4,
    budget_medium_weight: float = 0.6,
) -> tools.HallOfFame:
    hof = tools.HallOfFame(5)
    pop_size = len(population)
    n_workers = executor._max_workers if hasattr(executor, '_max_workers') else 1

    _immig_rate = max(1, int(pop_size * immigration_rate)) if immigration_rate > 0 else 0

    pending: Dict[cf.Future[Tuple[float, int]], int] = {}
    truly_evaluated: Set[Tuple[Edge, ...]] = set()
    for ind in population:
        if ind.fitness.values[0] != float("inf"):
            truly_evaluated.add(tuple(ind))
    stagnation = 0
    best_ever_err = float("inf")

    pbar = tqdm(total=ngen, desc="Evolving", unit="gen", dynamic_ncols=True)

    for gen in range(ngen):
        if gen % surrogate_retrain_interval == 0 and surrogate is not None and archive is not None and archive.size() >= surrogate_warmup_samples and not pretrained_loaded:
            if surrogate_type == "gnn":
                raw = archive.get_raw_samples()
                samples = [s[0] for s in raw]
                y_arr = np.array([s[1] for s in raw], dtype=np.float64)
                surrogate.fit(samples, y_arr)
            else:
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
            truly_evaluated.add(tuple(population[idx]))
            return idx

        def _surrogate_filter(individuals, top_frac, rand_frac):
            if not surrogate.is_ready():
                return set(range(len(individuals)))
            n = len(individuals)
            if n == 0:
                return set()
            if surrogate_type == "gnn":
                from nn_surrogate.data import build_predict_list
                data_list = build_predict_list([tuple(ind) for ind in individuals])
                ratios = surrogate.predict(data_list)
                target_ratio = target_pq[0] / target_pq[1]
                predictions = np.abs(target_ratio - ratios)
            else:
                from rf_surrogate.features import extract_features
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
                g = Graph.from_edges(list(ind))
                if i in eval_set:
                    ind.fitness.values = (float("inf"), len(g.nodes))
                else:
                    pred_err = max(0.0, float(predictions[i]))
                    ind.fitness.values = (pred_err, len(g.nodes))
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
        if potential_model is not None and potential_model.is_ready() and len(selected) > 0:
            from potential_net.data import build_predict_batch
            data_list = build_predict_batch(
                [tuple(ind) for ind in selected], target_pq
            )
            flow_pred, s_short, s_medium = potential_model.predict(data_list)
            target_ratio = target_pq[0] / target_pq[1]
            flow_score = np.exp(-np.abs(flow_pred - target_ratio) / 0.01)
            raw_potentials = flow_score * (budget_short_weight * s_short + budget_medium_weight * s_medium)
            p_min, p_max = raw_potentials.min(), raw_potentials.max()
            if p_max - p_min > 1e-8:
                potentials = (raw_potentials - p_min) / (p_max - p_min)
            else:
                potentials = np.ones_like(raw_potentials) * 0.5

            sorted_idx = np.argsort(raw_potentials)[::-1]
            n_top = max(1, int(len(selected) * 0.5))
            top_set = set(sorted_idx[:n_top].tolist())

            for i, ind in enumerate(selected):
                if i in top_set:
                    n_tries = mutation_budget_base + int(mutation_budget_max_extra * potentials[i])
                    best_ind = ind
                    best_score = raw_potentials[i]
                    for _ in range(n_tries):
                        variant, = toolbox.mutate(toolbox.clone(ind))
                        v_data = build_predict_batch([tuple(variant)], target_pq)
                        vf, vs, vm = potential_model.predict(v_data)
                        v_flow_score = np.exp(-abs(vf[0] - target_ratio) / 0.01)
                        v_score = v_flow_score * (budget_short_weight * vs[0] + budget_medium_weight * vm[0])
                        if v_score > best_score:
                            best_ind = variant
                            best_score = v_score
                    if best_ind is not ind:
                        selected[i] = creator.Individual(tuple(sorted(best_ind)))
                        del selected[i].fitness.values
                        n_mutated += 1
                else:
                    if random.random() < mutation_rate:
                        selected[i], = toolbox.mutate(ind)
                        del selected[i].fitness.values
                        n_mutated += 1
        else:
            for i, ind in enumerate(selected):
                if random.random() < mutation_rate:
                    selected[i], = toolbox.mutate(ind)
                    del selected[i].fitness.values
                    n_mutated += 1

        immigrants = [toolbox.individual() for _ in range(n_immigrants)]

        for i, ind in enumerate(selected):
            if len(tuple(ind)) > _cfg.max_edges:
                selected[i] = toolbox.individual()
        for i, ind in enumerate(immigrants):
            if len(tuple(ind)) > _cfg.max_edges:
                immigrants[i] = toolbox.individual()

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

        hof.update([ind for ind in ready if tuple(ind) in truly_evaluated])

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
            g = Graph.from_edges(list(ind))
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
                graph = Graph.from_edges(list(best_ind))
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
            for i, ind in enumerate(survivors):
                if len(tuple(ind)) > _cfg.max_edges:
                    survivors[i] = toolbox.individual()
            for i, ind in enumerate(new_indiv_list):
                if len(tuple(ind)) > _cfg.max_edges:
                    new_indiv_list[i] = toolbox.individual()
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
    surrogate_type: Optional[str] = None,
    surrogate_top_fraction: Optional[float] = None,
    surrogate_random_eval_fraction: Optional[float] = None,
    surrogate_warmup_samples: Optional[int] = None,
    surrogate_retrain_interval: Optional[int] = None,
    surrogate_n_estimators: Optional[int] = None,
    surrogate_max_depth: Optional[int] = None,
    potential_model_path: Optional[str] = None,
    mutation_budget_base: Optional[int] = None,
    mutation_budget_max_extra: Optional[int] = None,
    budget_short_weight: Optional[float] = None,
    budget_medium_weight: Optional[float] = None,
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
    surrogate_type = surrogate_type if surrogate_type is not None else _cfg.surrogate_type
    surrogate_top_fraction = surrogate_top_fraction if surrogate_top_fraction is not None else _cfg.surrogate_top_fraction
    surrogate_random_eval_fraction = surrogate_random_eval_fraction if surrogate_random_eval_fraction is not None else _cfg.surrogate_random_eval_fraction
    surrogate_warmup_samples = surrogate_warmup_samples if surrogate_warmup_samples is not None else _cfg.surrogate_warmup_samples
    surrogate_retrain_interval = surrogate_retrain_interval if surrogate_retrain_interval is not None else _cfg.surrogate_retrain_interval
    surrogate_n_estimators = surrogate_n_estimators if surrogate_n_estimators is not None else _cfg.surrogate_n_estimators
    surrogate_max_depth = surrogate_max_depth if surrogate_max_depth is not None else _cfg.surrogate_max_depth
    potential_model_path = potential_model_path if potential_model_path is not None else _cfg.potential_model_path
    mutation_budget_base = mutation_budget_base if mutation_budget_base is not None else _cfg.potential_mutation_budget_base
    mutation_budget_max_extra = mutation_budget_max_extra if mutation_budget_max_extra is not None else _cfg.potential_mutation_budget_max_extra
    budget_short_weight = budget_short_weight if budget_short_weight is not None else _cfg.potential_budget_short_weight
    budget_medium_weight = budget_medium_weight if budget_medium_weight is not None else _cfg.potential_budget_medium_weight

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
    toolbox.register("mate", partial(_crossover_wrapper, weights=tuple(_cfg.crossover_weights)))
    toolbox.register("mutate", partial(_mutate_wrapper, weights=mutation_weights, max_tries=_cfg.mutation_max_tries))

    print(f"\nGenerating initial population ({pop_size} individuals)...")
    population = toolbox.population(n=pop_size)

    if _cfg.seed_path is not None:
        import json
        with open(_cfg.seed_path, encoding="utf-8") as sf:
            seed_data = json.load(sf)
        seed_graphs = seed_data.get("collection", {})
        for i, entry in enumerate(seed_graphs.values()):
            if i >= pop_size:
                break
            seed_edges_list = [(e[0], e[1]) for e in entry["edges"]]
            seed_edges_tuple = tuple(sorted(seed_edges_list))
            population[i] = creator.Individual(seed_edges_tuple)
        print(f"  Seeded with {min(len(seed_graphs), pop_size)} known solution(s) from {_cfg.seed_path}")

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
        pretrained_loaded = False
        if surrogate_type is not None:
            from rf_surrogate.archive import SurrogateArchive
            archive = SurrogateArchive()
            for ind in population:
                fit = ind.fitness.values[0]
                if fit != float("inf"):
                    archive.add(tuple(ind), fit)
            if surrogate_type == "gnn":
                pretrained_path = "output/trained_gnn.pt"
                if pretrained_loaded := os.path.exists(pretrained_path):
                    from nn_surrogate.model import SurrogateGNN
                    surrogate = SurrogateGNN()
                    surrogate.load(pretrained_path)
                    print(f"  Loaded pre-trained GNN: {pretrained_path}")
                else:
                    print(f"  No pre-trained GNN at {pretrained_path}")
                    print(f"  Run: python tools/train_gnn.py")
            elif archive.size() >= surrogate_warmup_samples:
                from rf_surrogate.model import SurrogateRF
                surrogate = SurrogateRF(
                    n_estimators=surrogate_n_estimators,
                    max_depth=surrogate_max_depth,
                )
                X, y = archive.get_data()
                surrogate.fit(X, y)

        potential_model = None
        if potential_model_path is not None and os.path.exists(potential_model_path):
            from potential_net.model import PotentialNet
            potential_model = PotentialNet()
            potential_model.load(potential_model_path)
            print(f"  Loaded potential model: {potential_model_path}")

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
            surrogate_type=surrogate_type,
            target_pq=target_pq,
            pretrained_loaded=pretrained_loaded,
            potential_model=potential_model,
            mutation_budget_base=mutation_budget_base,
            mutation_budget_max_extra=mutation_budget_max_extra,
            budget_short_weight=budget_short_weight,
            budget_medium_weight=budget_medium_weight,
        )
        elapsed = time.perf_counter() - t_start

    results = []
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed/60:.1f}min)")

    for rank, ind in enumerate(hof):
        graph = Graph.from_edges(list(ind))
        real_err, real_nodes, _ = evaluate_cached(
            tuple(ind), target_pq, threads=threads,
            max_denominator=max_denominator,
            mode=mode,  # type: ignore[arg-type]
        )
        err = real_err if real_err != float("inf") else ind.fitness.values[0]
        nodes = real_nodes if real_nodes > 0 else ind.fitness.values[1]

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
    out_path = output_path or os.path.join("output", "ga_top5.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")

    if surrogate is not None and surrogate.is_ready() and archive is not None:
        os.makedirs("output", exist_ok=True)
        if surrogate_type == "gnn":
            from rf_surrogate.features import extract_features
            surrogate.save("output/gnn_model.pt")
            raw = archive.get_raw_samples()
            X_arr = np.array(
                [extract_features(s[0]) for s in raw], dtype=np.float64
            )
            y_arr = np.array([s[1] for s in raw], dtype=np.float64)
            np.save("output/gnn_archive_X.npy", X_arr)
            np.save("output/gnn_archive_y.npy", y_arr)
            import pickle
            with open("output/gnn_samples.pkl", "wb") as fs:
                pickle.dump(raw, fs)
            print(f"  GNN model & archive saved to output/  (n_samples={archive.size()})")
        else:
            import pickle
            X, y = archive.get_data()
            np.save("output/rf_archive_X.npy", X)
            np.save("output/rf_archive_y.npy", y)
            with open("output/rf_model.pkl", "wb") as f:
                pickle.dump(surrogate._model, f)
            print(f"  RF model & archive saved to output/  (n_samples={archive.size()})")

    history.to_json(_cfg.history_path)
    print(f"\n{history.summary()}")

    return results
