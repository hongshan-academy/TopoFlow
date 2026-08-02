import random
from graph import Graph
from ga.mutation import mutate_concat, MUTATION_FNS
from ga.generation import generate_strict_graph
from ga.fitness import evaluate_cached
from config import DEFAULT_CONFIG as cfg

random.seed(42)

print(f"MUTATION_FNS: {[f.__name__ for f in MUTATION_FNS]}")
print(f"weights len: {len(cfg.mutation_weights)} (expect 5)")
print()

# Test 1: Generate random graphs, concat
for i in range(5):
    g = generate_strict_graph(random.randint(10, 30))
    if not g.is_valid(strict=True) or len(g.edges) > cfg.max_edges:
        continue

    err_before, nodes_before = evaluate_cached(
        tuple(sorted(g.edges)), (325, 799), mode="mixed"
    )
    print(f"Test {i}: src={len(g.nodes)}n/{len(g.edges)}e err={err_before:.6f}")

    result = mutate_concat(g)
    if result is None:
        print(f"  concat returned None")
        continue

    err_after, nodes_after = evaluate_cached(
        tuple(sorted(result.edges)), (325, 799), mode="mixed"
    )
    print(
        f"  concat={len(result.nodes)}n/{len(result.edges)}e "
        f"err={err_after:.6f} valid={result.is_valid(strict=True)}"
    )

print()

# Test 2: Concat from known solution
known = tuple(
    sorted(
        [
            ("In", "C3_1"),
            ("S2_0", "C3_0"),
            ("S2_0", "C3_2"),
            ("S2_1", "C3_6"),
            ("S2_1", "S3_5"),
            ("S2_2", "C3_0"),
            ("S2_2", "C3_7"),
            ("S3_0", "C2_0"),
            ("S3_0", "C3_0"),
            ("S3_0", "Out"),
            ("S3_1", "C3_2"),
            ("S3_1", "C3_4"),
            ("S3_1", "S2_0"),
            ("S3_2", "C3_3"),
            ("S3_2", "C3_3"),
            ("S3_2", "S3_4"),
            ("S3_3", "C3_4"),
            ("S3_3", "C3_4"),
            ("S3_3", "S3_2"),
            ("S3_4", "C3_5"),
            ("S3_4", "C3_5"),
            ("S3_4", "S3_1"),
            ("S3_5", "C3_6"),
            ("S3_5", "C3_6"),
            ("S3_5", "C3_7"),
            ("S3_6", "C3_5"),
            ("S3_6", "S2_1"),
            ("S3_6", "S2_2"),
            ("C2_0", "S3_6"),
            ("C3_0", "C3_7"),
            ("C3_1", "S3_0"),
            ("C3_2", "C3_1"),
            ("C3_3", "C3_2"),
            ("C3_4", "C3_3"),
            ("C3_5", "S3_3"),
            ("C3_6", "C2_0"),
            ("C3_7", "C3_1"),
        ]
    )
)
g_known = Graph.from_edges(list(known))
known_err = evaluate_cached(known, (325, 799), mode="mixed")[0]
print(f"Known solution: err={known_err:.6f}")

for i in range(5):
    result2 = mutate_concat(g_known)
    if result2 is not None:
        err2, n2 = evaluate_cached(
            tuple(sorted(result2.edges)), (325, 799), mode="mixed"
        )
        print(
            f"  concat from known #{i}: "
            f"{len(result2.nodes)}n/{len(result2.edges)}e "
            f"err={err2:.10f} valid={result2.is_valid(strict=True)}"
        )
    else:
        print(f"  concat from known #{i}: returned None")
