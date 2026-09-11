"""Splitter builders: split schemes and split trees.

Combines the original ``chaifen.py`` helpers (``split_node`` /
``can_partition`` / ``partition_groups`` / ``bfs_min_splits``) with the
ratio-scheme helpers used by ``server.py`` (``find_ratio_scheme`` /
``split_leaves``).
"""

from __future__ import annotations
from collections import deque

# ===== Configurable parameters (used by the original chaifen.py tool) =====
INIT_VALUE = 1536             # initial value
TARGETS = (423, 829, 284)     # sums of the three target groups
MAX_LEAVES = 20               # pruning cap on the number of leaves


def split_node(v: int) -> list[tuple[int, ...]]:
    """Return the children of ``v`` for an even/odd split (2-way or 3-way)."""
    res: list[tuple[int, ...]] = []
    if v % 2 == 0:
        res.append((v // 2, v // 2))
    if v % 3 == 0:
        res.append((v // 3, v // 3, v // 3))
    return res


def can_partition(counts: dict[int, int], targets: tuple[int, int, int]) -> bool:
    """Check whether multiset ``counts`` splits into three groups of ``targets``.

    Grouping knapsack DP over integer bitsets: ``dp[s1]`` is a bitset whose
    ``s2``-th bit means the state ``(group1=s1, group2=s2)`` is reachable.
    Each leaf goes to exactly one group; group 3 takes the remainder. The bit
    arithmetic runs in C, so it is much faster than set-based DP.
    """
    t1, t2, t3 = targets
    leaves = [v for v, c in counts.items() for _ in range(c)]
    if sum(leaves) != t1 + t2 + t3:
        return False

    mask2 = (1 << (t2 + 1)) - 1     # upper-bound bitmask for group-2 sum
    dp = [0] * (t1 + 1)
    dp[0] = 1                       # (0, 0) reachable
    for v in leaves:
        ndp = dp[:]                 # leaf to group 3: state unchanged
        for s1 in range(t1 + 1):
            cur = dp[s1]
            if cur:
                ndp[s1] |= (cur << v) & mask2        # to group 2
                if s1 + v <= t1:
                    ndp[s1 + v] |= cur               # to group 1
        dp = ndp
    return bool(dp[t1] & (1 << t2))


def partition_groups(
    counts: dict[int, int],
    targets: tuple[int, int, int],
) -> tuple[list[int], list[int], list[int]] | None:
    """Split ``counts`` into three groups of sums ``targets``.

    Returns the actual groups, or ``None`` if impossible. Uses a bounded
    grouping-knapsack DP and backtracks the concrete assignment.
    ``counts`` maps value -> multiplicity.
    """
    t1, t2, t3 = targets
    leaves = []
    for v, c in sorted(counts.items(), reverse=True):
        leaves.extend([v] * c)
    if sum(leaves) != t1 + t2 + t3:
        return None

    layers = [{(0, 0)}]
    for v in leaves:
        cur = set()
        for s1, s2 in layers[-1]:
            if s1 + v <= t1:
                cur.add((s1 + v, s2))   # to group 1
            if s2 + v <= t2:
                cur.add((s1, s2 + v))   # to group 2
            cur.add((s1, s2))           # to group 3: state unchanged
        if not cur:
            return None
        layers.append(cur)

    if (t1, t2) not in layers[-1]:
        return None

    g1, g2, g3 = [], [], []
    s1, s2 = t1, t2
    for i in range(len(leaves) - 1, -1, -1):
        v = leaves[i]
        if (s1 - v, s2) in layers[i]:
            g1.append(v)
            s1 -= v
        elif (s1, s2 - v) in layers[i]:
            g2.append(v)
            s2 -= v
        else:
            g3.append(v)
    return g1, g2, g3


def bfs_min_splits(
    init: int,
    targets: tuple[int, int, int],
    max_leaves: int,
) -> tuple[tuple[list[int], list[int], list[int]], int, list[tuple[int, tuple[int, ...]]]] | None:
    """BFS for the fewest splits; returns ``(groups, steps, path)``."""
    start = {init: 1}
    visited = {tuple(sorted(start.items()))}
    queue: deque[tuple[dict[int, int], int, list[tuple[int, tuple[int, ...]]]]] = deque(
        [(start, 0, [])]
    )

    while queue:
        counts, steps, path = queue.popleft()
        if sum(counts.values()) > max_leaves:
            continue
        if can_partition(counts, targets):
            groups = partition_groups(counts, targets)
            if groups is None:
                return None
            return groups, steps, path

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
                    queue.append((newc, steps + 1, path + [(v, children)]))
    return None


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
        if can_partition(counts, targets) and (cap is None or max(counts) < cap):
            groups = partition_groups(counts, targets)
            if groups is None:
                return None
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


if __name__ == "__main__":
    result = bfs_min_splits(INIT_VALUE, TARGETS, MAX_LEAVES)
    if result:
        groups, steps, path = result
        print(f"min splits: {steps}")
        print("split process:")
        for parent, children in path:
            print(f"  {parent} -> {children}")
        print("\ngroup check:")
        for name, g in zip(TARGETS, groups):
            print(f"  target {name}: {g} (sum = {sum(g)})")
    else:
        print("no solution")
