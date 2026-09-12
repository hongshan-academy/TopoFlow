"""Splitter builders: split schemes and split trees.

Combines the original ``chaifen.py`` helpers (``split_node`` /
``partition_groups``) with the ratio-scheme helpers used by ``server.py``
(``find_ratio_scheme`` / ``split_leaves``).
"""

from __future__ import annotations
from collections import deque


def split_node(v: int) -> list[tuple[int, ...]]:
    """Return the children of ``v`` for an even/odd split (2-way or 3-way)."""
    res: list[tuple[int, ...]] = []
    if v % 2 == 0:
        res.append((v // 2, v // 2))
    if v % 3 == 0:
        res.append((v // 3, v // 3, v // 3))
    return res


def partition_groups(
    counts: dict[int, int],
    targets: tuple[int, int, int],
) -> tuple[list[int], list[int], list[int]] | None:
    """Split ``counts`` into three groups of sums ``targets``.

    Returns the actual groups, or ``None`` if impossible. The grouping
    knapsack DP keeps, for every group-1 sum ``s1``, a bitset whose ``s2``-th
    bit means ``(group1=s1, group2=s2)`` is reachable; the bit arithmetic runs
    in C and the layers are backtracked into a concrete assignment.
    ``counts`` maps value -> multiplicity.
    """
    t1, t2, t3 = targets
    leaves: list[int] = []
    for v, c in sorted(counts.items(), reverse=True):
        leaves.extend([v] * c)
    if sum(leaves) != t1 + t2 + t3:
        return None

    mask2 = (1 << (t2 + 1)) - 1
    layers: list[list[int]] = [[0] * (t1 + 1)]
    layers[0][0] = 1
    for v in leaves:
        prev = layers[-1]
        cur = prev[:]
        for s1 in range(t1 + 1):
            bits = prev[s1]
            if bits:
                cur[s1] |= (bits << v) & mask2
                if s1 + v <= t1:
                    cur[s1 + v] |= bits
        layers.append(cur)

    if not (layers[-1][t1] & (1 << t2)):
        return None

    g1, g2, g3 = [], [], []
    s1, s2 = t1, t2
    for i in range(len(leaves) - 1, -1, -1):
        v = leaves[i]
        prev = layers[i]
        if s1 - v >= 0 and (prev[s1 - v] >> s2) & 1:
            g1.append(v)
            s1 -= v
        elif s2 - v >= 0 and (prev[s1] >> (s2 - v)) & 1:
            g2.append(v)
            s2 -= v
        else:
            g3.append(v)
    return g1, g2, g3


def find_ratio_scheme(p: int, q: int, max_share: float | None = None) -> tuple[int, int, int]:
    """Find the smallest ``a + 1.85b`` with ``N = 2^a * 3^b >= q``.

    Returns ``(a, b, N)``. When ``max_share`` is given (a share-count
    constraint) it additionally requires ``N > 1 / max_share`` so that any
    large share can be split further down to ``1/N < max_share``; the scheme
    still keeps ``N`` minimal. Whether further splitting is actually needed is
    decided by :func:`split_leaves`.
    """
    best = None
    max_a = max(0, q.bit_length()) + 1
    for b in range(0, 21):
        for a in range(0, max_a + 1):
            N = (1 << a) * (3 ** b)
            if N < q:
                continue
            if max_share is not None and N <= 1.0 / max_share:
                continue
            score = a + 1.85 * b
            if best is None or score < best[0] or (score == best[0] and N < best[3]):
                best = (score, a, b, N)
    if best is None:
        raise ValueError(f"no a, b satisfy N=2^a*3^b>=q (q={q})")
    return best[1], best[2], best[3]


def split_leaves(N: int, targets: tuple[int, int, int],
                 max_share: float | None, max_leaves: int,
                 ) -> tuple[tuple[list[int], list[int], list[int]], list[tuple[int, tuple[int, ...]]]] | None:
    """Repeatedly split ``N`` and return ``(groups, path)``.

    Requires the leaf multiset to split into three groups of sums ``targets``
    and, when ``max_share`` is given, every leaf value strictly below
    ``max_share * N``. Prefers the fewest splits; returns ``None`` if no valid
    decomposition exists.
    """
    cap = max_share * N if max_share is not None else None
    start = {N: 1}
    queue: deque[tuple[dict[int, int], list[tuple[int, tuple[int, ...]]]]] = deque([(start, [])])
    visited = {tuple(sorted(start.items()))}
    while queue:
        counts, path = queue.popleft()
        if sum(counts.values()) > max_leaves:
            continue
        if cap is None or max(counts) < cap:
            groups = partition_groups(counts, targets)
            if groups is not None:
                return groups, path
        for v in list(counts.keys()):
            for children in split_node(v):
                newc = dict(counts)
                if newc[v] == 1:
                    del newc[v]
                else:
                    newc[v] -= 1
                for c in children:
                    newc[c] = newc.get(c, 0) + 1
                key = tuple(sorted(newc.items()))
                if key not in visited:
                    visited.add(key)
                    queue.append((newc, path + [(v, children)]))
    return None
