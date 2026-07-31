import bisect
from typing import List, Tuple

from ga.chromosome import (
    decode, port_count, _make_out_ports, _make_in_ports,
)

FEATURE_NAMES = [
    "s2", "s3", "c2", "c3",
    "total_nodes", "total_ports", "balance_left", "balance_right",
    "inv_ratio", "fixed_pts_ratio", "cycle_cnt_ratio",
    "longest_cycle_ratio", "lis_ratio", "is_derangement",
    "edges_per_port", "self_loops_per_port", "is_valid",
    "avg_out_deg", "max_out_deg", "avg_in_deg", "max_in_deg",
]

FEATURE_DIM = len(FEATURE_NAMES)


def _inversions(perm: Tuple[int, ...]) -> int:
    n = len(perm)
    if n <= 1:
        return 0
    count = 0
    for i in range(n):
        pi = perm[i]
        for j in range(i + 1, n):
            if pi > perm[j]:
                count += 1
    return count


def _inversions_ratio(perm: Tuple[int, ...]) -> float:
    n = len(perm)
    if n <= 1:
        return 0.0
    return _inversions(perm) / (n * (n - 1) / 2)


def _fixed_points_ratio(perm: Tuple[int, ...]) -> float:
    n = len(perm)
    if n == 0:
        return 0.0
    return sum(1 for i, p in enumerate(perm) if i == p) / n


def _cycles(perm: Tuple[int, ...]) -> List[int]:
    n = len(perm)
    visited = [False] * n
    lengths: List[int] = []
    for i in range(n):
        if not visited[i]:
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                length += 1
            lengths.append(length)
    return lengths


def _cycle_count_ratio(perm: Tuple[int, ...]) -> float:
    n = len(perm)
    if n == 0:
        return 0.0
    return len(_cycles(perm)) / n


def _longest_cycle_ratio(perm: Tuple[int, ...]) -> float:
    n = len(perm)
    if n == 0:
        return 0.0
    return max(_cycles(perm)) / n


def _lis_length(perm: Tuple[int, ...]) -> int:
    tails: List[int] = []
    for x in perm:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _lis_ratio(perm: Tuple[int, ...]) -> float:
    n = len(perm)
    if n == 0:
        return 1.0
    return _lis_length(perm) / n


def _is_derangement(perm: Tuple[int, ...]) -> float:
    return 1.0 if all(i != p for i, p in enumerate(perm)) else 0.0


def extract_features(chromosome: Tuple[int, ...]) -> List[float]:
    s2, s3, c2, c3 = int(chromosome[0]), int(chromosome[1]), int(chromosome[2]), int(chromosome[3])
    perm = tuple(int(x) for x in chromosome[4:])

    n_ports = len(perm)

    total_nodes = 2 + s2 + s3 + c2 + c3
    balance_left = s2 + 2 * s3
    balance_right = c2 + 2 * c3

    perm_features = [
        _inversions_ratio(perm),
        _fixed_points_ratio(perm),
        _cycle_count_ratio(perm),
        _longest_cycle_ratio(perm),
        _lis_ratio(perm),
        _is_derangement(perm),
    ]

    graph = decode(chromosome)
    edges = len(graph.edges)

    edges_per_port = edges / n_ports if n_ports > 0 else 0.0

    out_ports = _make_out_ports(s2, s3, c2, c3)
    in_ports = _make_in_ports(s2, s3, c2, c3)
    self_loops = 0
    for out_idx, in_idx in enumerate(perm):
        if out_ports[out_idx][0] == in_ports[in_idx][0]:
            self_loops += 1
    self_loops_per_port = self_loops / n_ports if n_ports > 0 else 0.0

    is_valid = 1.0 if graph.is_valid() else 0.0

    internal_out: List[int] = []
    internal_in: List[int] = []
    for node in graph.nodes:
        if node in ("In", "Out"):
            continue
        in_d, out_d = graph.degrees[node]
        internal_in.append(in_d)
        internal_out.append(out_d)

    if internal_out:
        avg_out = sum(internal_out) / len(internal_out)
        max_out = max(internal_out) / 3.0
    else:
        avg_out = 0.0
        max_out = 0.0

    if internal_in:
        avg_in = sum(internal_in) / len(internal_in)
        max_in = max(internal_in) / 3.0
    else:
        avg_in = 0.0
        max_in = 0.0

    return [
        float(s2), float(s3), float(c2), float(c3),
        float(total_nodes), float(n_ports),
        float(balance_left), float(balance_right),
        *perm_features,
        edges_per_port, self_loops_per_port, is_valid,
        avg_out / 3.0, max_out, avg_in / 3.0, max_in,
    ]
