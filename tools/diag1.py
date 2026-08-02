import random, math, json, time
from collections import Counter, defaultdict
from fractions import Fraction
from simulator import TopoFlowSimulator
from graph import Graph
from ga.generation import generate_strict_graph
from config import DEFAULT_CONFIG as cfg

random.seed(42)
N_SAMPLES = 300
results = []
target = Fraction(325, 799)


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


print(f"Generating {N_SAMPLES} random graphs and simulating...")
t0 = time.perf_counter()
for i in range(N_SAMPLES):
    if i % 30 == 0:
        print(f"  {i}/{N_SAMPLES} ({time.perf_counter() - t0:.1f}s)")
    n_internal = random.choices(
        cfg.internal_nodes_choices, weights=cfg.internal_nodes_weights, k=1
    )[0]
    g = generate_strict_graph(n_internal)
    if not g.is_valid(strict=True) or len(g.edges) > cfg.max_edges:
        continue
    sim = TopoFlowSimulator(g)
    try:
        cycle = sim.run_until_cycle(max_frames=cfg.sim_max_frames)
    except Exception:
        cycle = {"converged": False}
    if not cycle.get("converged", False):
        continue

    period = cycle["period"]
    if period <= 0:
        continue

    # Find source edge (In -> first_target)
    source_target = g.out_edges["In"][0]
    source_num = 0
    for edge_key, (num, _) in cycle["edge_ratios"].items():
        if edge_key[0] == "In" and edge_key[1] == source_target[1]:
            source_num = num
            break

    flow_frac = Fraction(source_num, period) if period > 0 else Fraction(0, 1)
    limited = Fraction(float(flow_frac)).limit_denominator(10000)
    err = float(abs(target - limited))

    results.append(
        {
            "nodes": len(g.nodes),
            "edges": len(g.edges),
            "period": period,
            "num": source_num,
            "flow_num": flow_frac.numerator,
            "flow_den": flow_frac.denominator,
            "flow_float": float(flow_frac),
            "error": err,
            "limited_num": limited.numerator,
            "limited_den": limited.denominator,
        }
    )

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s. Valid+converged: {len(results)}")

# ── Period analysis ──
periods = Counter()
for r in results:
    periods[r["period"]] += 1

period_primes = Counter()
for r in results:
    for p in prime_factors(r["period"]):
        period_primes[p] += 1

print(f"\n=== Period distribution (top 25) ===")
for period, count in periods.most_common(25):
    pf = prime_factors(period)
    pf_str = "x".join(
        f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(pf.items())
    )
    print(f"  period={period:>5} (={pf_str}) count={count:>3}")

print(f"\n=== Period prime factor frequencies ===")
for prime, count in sorted(period_primes.items()):
    frac = count / len(results)
    bar = "#" * int(frac * 50)
    print(f"  p={prime:>3}: {count:>4}/{len(results)} ({frac*100:5.1f}%) {bar}")

# ── Target-specific ──
print(f"\n=== Target: 325/799 ===")
print(f"799 prime factors: {dict(prime_factors(799))}")
print(f"Has factors 17={17 in period_primes}, 47={47 in period_primes}")
periods_with_799 = [(p, c) for p, c in periods.items() if p % 799 == 0]
print(f"Periods multiple of 799: {periods_with_799}")
print(f"Period 799: appears {periods.get(799, 0)} times")

# ── What flow ratios appear? ──
flow_counts = Counter()
for r in results:
    flow_counts[(r["flow_num"], r["flow_den"])] += 1

target_reached = flow_counts.get((325, 799), 0)
close = sum(
    1
    for (n, d), c in flow_counts.items()
    if abs(float(Fraction(n, d)) - float(target)) < 0.01
)
print(f"\n=== Flow ratios ===")
print(f"Exact target (325/799): {target_reached}")
print(f"Within 0.01 of target: {close}")

# ── Error distribution ──
print(f"\n=== Error distribution ===")
errs = [r["error"] for r in results]
for threshold in [0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]:
    count = sum(1 for e in errs if e <= threshold)
    print(
        f"  err <= {threshold:.0e}: {count}/{len(results)} ({count/len(results)*100:.1f}%)"
    )

# ── Limited denominator distribution ──
limited_dens = Counter()
for r in results:
    limited_dens[r["limited_den"]] += 1
print(f"\n=== Limited denominator dist (top 15) ===")
for den, count in limited_dens.most_common(15):
    pf = prime_factors(den)
    pf_str = "x".join(
        f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(pf.items())
    )
    print(f"  den={den:>5} (={pf_str}) count={count:>3}")

# ── Top 10 closest ──
sorted_by_err = sorted(results, key=lambda r: r["error"])
print(f"\n=== Top 10 closest to target ===")
for r in sorted_by_err[:10]:
    flow_str = f'{r["flow_num"]}/{r["flow_den"]}'
    print(
        f'  err={r["error"]:.2e}  flow={flow_str}  period={r["period"]}  edges={r["edges"]}'
    )

# ── Save ──
with open("output/diag1_results.json", "w") as f:
    json.dump({"results": results}, f, indent=2)
print("\nSaved to output/diag1_results.json")
