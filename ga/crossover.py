import random
from typing import Dict, Set, Tuple


def _pmx_perm(perm1: Tuple[int, ...], perm2: Tuple[int, ...]) -> Tuple[int, ...]:
    n = len(perm1)
    if n < 2:
        return perm1
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n

    for i in range(a, b):
        child[i] = perm1[i]

    mapping: Dict[int, int] = {}
    for i in range(a, b):
        mapping.setdefault(perm1[i], perm2[i])
        mapping.setdefault(perm2[i], perm1[i])

    for i in range(n):
        if a <= i < b:
            continue
        val = perm2[i]
        visited: Set[int] = set()
        while val in child and val not in visited:
            visited.add(val)
            if val in mapping:
                val = mapping[val]
            else:
                break
        child[i] = val

    return tuple(child)


def crossover_pmx(ind1: Tuple[int, ...], ind2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    s2_a, s3_a, c2_a, c3_a = ind1[:4]
    s2_b, s3_b, c2_b, c3_b = ind2[:4]
    perm_a = ind1[4:]
    perm_b = ind2[4:]

    counts_a = (s2_a, s3_a, c2_a, c3_a)
    counts_b = (s2_b, s3_b, c2_b, c3_b)

    if counts_a == counts_b:
        child1 = counts_a + _pmx_perm(perm_a, perm_b)
        child2 = counts_a + _pmx_perm(perm_b, perm_a)
        return child1, child2

    if random.random() < 0.5:
        c1_counts = counts_a
        c2_counts = counts_b
        c1_base = list(perm_a)
        c2_base = list(perm_b)
    else:
        c1_counts = counts_b
        c2_counts = counts_a
        c1_base = list(perm_b)
        c2_base = list(perm_a)

    if len(c1_base) >= 2:
        i, j = random.sample(range(len(c1_base)), 2)
        c1_base[i], c1_base[j] = c1_base[j], c1_base[i]

    if len(c2_base) >= 2:
        i, j = random.sample(range(len(c2_base)), 2)
        c2_base[i], c2_base[j] = c2_base[j], c2_base[i]

    return (
        c1_counts + tuple(c1_base),
        c2_counts + tuple(c2_base),
    )
