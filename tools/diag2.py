import json, time
from collections import Counter
from fractions import Fraction
from simulator import TopoFlowSimulator
from graph import Graph
from config import DEFAULT_CONFIG as cfg


def prime_factors(n):
    factors = Counter()
    d = 2
    m = n
    while d * d <= m:
        while m % d == 0:
            factors[d] += 1
            m //= d
        d += 1 if d == 2 else 2
    if m > 1:
        factors[m] += 1
    return factors


target = Fraction(325, 799)


def analyze_best_chain(history_path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    with open(history_path) as f:
        h = json.load(f)

    chain = h.get("best_chain", [])
    print(f"Best chain steps: {len(chain)}")

    for step in chain:
        gen = step["gen"]
        err = step["fitness"][0]
        edges_raw = step.get("edges", [])
        if not edges_raw:
            continue

        edges = [(e[0], e[1]) for e in edges_raw]
        g = Graph.from_edges(edges)
        if not g.is_valid(strict=True):
            print(f"  gen={gen:>5}: err={err:.2e} | INVALID GRAPH")
            continue

        sim = TopoFlowSimulator(g)
        try:
            cycle = sim.run_until_cycle(max_frames=cfg.sim_max_frames)
        except Exception:
            cycle = {"converged": False}

        if not cycle.get("converged", False):
            print(f"  gen={gen:>5}: err={err:.2e} | {len(g.nodes)}n/{len(g.edges)}e | SIM DID NOT CONVERGE")
            continue

        period = cycle["period"]
        source_target = g.out_edges["In"][0]
        source_num = 0
        for edge_key, (num, _) in cycle["edge_ratios"].items():
            if edge_key[0] == "In" and edge_key[1] == source_target[1]:
                source_num = num
                break

        flow_frac = Fraction(source_num, period)
        limited = Fraction(float(flow_frac)).limit_denominator(10000)
        actual_err = float(abs(target - limited))

        pf = prime_factors(period)
        pf_str = "x".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(pf.items()))

        print(
            f"  gen={gen:>5}: recorded_err={err:.2e}  actual_err={actual_err:.2e}  "
            f"flow={flow_frac.numerator}/{flow_frac.denominator}  "
            f"period={period} ({pf_str})  "
            f"{len(g.nodes)}n/{len(g.edges)}e"
        )


def analyze_full_population(history_path, label):
    print(f"\n{'='*60}")
    print(f"  {label} — Full generation stats")
    print(f"{'='*60}")

    with open(history_path) as f:
        h = json.load(f)

    gens = h.get("generations", [])
    n_gens = len(gens)
    print(f"Total generations: {n_gens}")

    # Sample every 50th gen to analyze best individual
    periods = Counter()
    all_periods = []
    for gen_data in gens:
        if gen_data["gen"] % 50 != 0:
            continue
        edges_raw = gen_data.get("best_edges")
        if not edges_raw:
            continue
        edges = [(e[0], e[1]) for e in edges_raw]
        g = Graph.from_edges(edges)
        if not g.is_valid(strict=True):
            continue
        sim = TopoFlowSimulator(g)
        try:
            cycle = sim.run_until_cycle(max_frames=cfg.sim_max_frames)
        except Exception:
            continue
        if not cycle.get("converged", False):
            continue
        period = cycle["period"]
        periods[period] += 1
        all_periods.append(period)

    if not all_periods:
        print("  No converged samples found!")
        return

    print(f"Sampled {len(all_periods)} generations with converged sim")
    print(f"Unique periods: {len(periods)}")

    # Prime factor distribution
    period_primes = Counter()
    for p in all_periods:
        for pf in prime_factors(p):
            period_primes[pf] += 1

    print(f"\nPeriod prime factor frequencies:")
    for prime, count in sorted(period_primes.items()):
        frac = count / len(all_periods)
        bar = "#" * int(frac * 50)
        print(f"  p={prime:>3}: {count:>4}/{len(all_periods)} ({frac*100:5.1f}%) {bar}")

    # Check target primes
    print(f"\nTarget primes: 17 ({period_primes.get(17, 0)}), 47 ({period_primes.get(47, 0)})")
    periods_with_799 = [p for p in all_periods if p % 799 == 0]
    print(f"Periods multiple of 799: {periods_with_799}")


# ── Run analysis ──
print("Analyzing GA-converged populations...")
t0 = time.perf_counter()

analyze_best_chain("output/run1/ga_history.json", "Run 1 (max=50, pop=200, gen=600)")
analyze_best_chain("output/run2/ga_history.json", "Run 2 (max=40, pop=200, gen=1500)")
analyze_best_chain("output/run3/ga_history.json", "Run 3 (max=50, pop=200, gen=700)")
analyze_best_chain("output/run4/ga_history.json", "Run 4 (max=40, pop=200, gen=3000)")
analyze_best_chain("output/run5/ga_history.json", "Run 5 (max=40, pop=200, gen=2000, SEEDED)")

analyze_full_population("output/run2/ga_history.json", "Run 2 — Population periods over time")
analyze_full_population("output/run4/ga_history.json", "Run 4 — Population periods over time")
analyze_full_population("output/run5/ga_history.json", "Run 5 — Population periods over time")

elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed:.1f}s")
