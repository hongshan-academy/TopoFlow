import random
from typing import List, Tuple

from ga.chromosome import decode, encode, get_counts, get_perm, port_count, balance_counts, _make_out_ports, _make_in_ports

DELETE_MUTATION_TRIES = 5


def mutate_counts(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    s2, s3, c2, c3 = ind[:4]
    perm = ind[4:]
    n_old = port_count(s2, s3, c2, c3)

    field = random.choice(["s2", "s3", "c2", "c3"])
    delta = random.choice([-1, 1, 2])

    if field == "s2":
        s2 = max(0, s2 + delta)
    elif field == "s3":
        s3 = max(0, s3 + delta)
    elif field == "c2":
        c2 = max(0, c2 + delta)
    else:
        c3 = max(0, c3 + delta)

    s2, s3, c2, c3 = balance_counts(s2, s3, c2, c3)
    s2, s3, c2, c3 = max(0, s2), max(0, s3), max(0, c2), max(0, c3)

    n_new = port_count(s2, s3, c2, c3)

    if n_new == n_old:
        new_perm: Tuple[int, ...] = perm
    else:
        new_perm_list: List[int] = list(range(n_new))
        random.shuffle(new_perm_list)
        new_perm = tuple(new_perm_list)

    return (s2, s3, c2, c3) + new_perm


def mutate_counts_preserve(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    s2, s3, c2, c3 = ind[:4]
    perm = ind[4:]
    n_old = port_count(s2, s3, c2, c3)

    field = random.choice(["s2", "s3", "c2", "c3"])
    delta = random.choice([-1, 1, 2])

    if field == "s2":
        s2 = max(0, s2 + delta)
    elif field == "s3":
        s3 = max(0, s3 + delta)
    elif field == "c2":
        c2 = max(0, c2 + delta)
    else:
        c3 = max(0, c3 + delta)

    s2, s3, c2, c3 = balance_counts(s2, s3, c2, c3)
    s2, s3, c2, c3 = max(0, s2), max(0, s3), max(0, c2), max(0, c3)

    n_new = port_count(s2, s3, c2, c3)

    if n_new == n_old:
        return (s2, s3, c2, c3) + perm

    out_old = _make_out_ports(ind[0], ind[1], ind[2], ind[3])
    in_old = _make_in_ports(ind[0], ind[1], ind[2], ind[3])
    out_new = _make_out_ports(s2, s3, c2, c3)
    in_new = _make_in_ports(s2, s3, c2, c3)

    out_map = {}
    for old_idx, (node, port) in enumerate(out_old):
        for new_idx, (n2, p2) in enumerate(out_new):
            if node == n2 and port == p2:
                out_map[old_idx] = new_idx
                break

    in_map = {}
    for old_idx, (node, port) in enumerate(in_old):
        for new_idx, (n2, p2) in enumerate(in_new):
            if node == n2 and port == p2:
                in_map[old_idx] = new_idx
                break

    new_perm = [-1] * n_new
    unassigned_in = set(range(n_new))

    for old_i, old_j in enumerate(perm):
        if old_i in out_map and old_j in in_map:
            new_i = out_map[old_i]
            new_j = in_map[old_j]
            if out_new[new_i][0] != in_new[new_j][0]:
                new_perm[new_i] = new_j
                unassigned_in.discard(new_j)

    unassigned_out = [i for i in range(n_new) if new_perm[i] == -1]
    shuffled_in = list(unassigned_in)
    random.shuffle(shuffled_in)

    for i, j in zip(unassigned_out, shuffled_in):
        new_perm[i] = j

    return (s2, s3, c2, c3) + tuple(new_perm)


def mutate_perm_swap(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    s2, s3, c2, c3 = ind[:4]
    perm = list(ind[4:])
    if len(perm) < 2:
        return ind
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return (s2, s3, c2, c3) + tuple(perm)


def mutate_perm_scramble(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    s2, s3, c2, c3 = ind[:4]
    perm = list(ind[4:])
    if len(perm) < 3:
        return mutate_perm_swap(ind)
    a, b = sorted(random.sample(range(len(perm)), 2))
    segment = perm[a:b + 1]
    random.shuffle(segment)
    perm[a:b + 1] = segment
    return (s2, s3, c2, c3) + tuple(perm)


def mutate_perm_reverse(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    s2, s3, c2, c3 = ind[:4]
    perm = list(ind[4:])
    if len(perm) < 3:
        return mutate_perm_swap(ind)
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm[a:b + 1] = perm[a:b + 1][::-1]
    return (s2, s3, c2, c3) + tuple(perm)


def mutate_graph_delete(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    g = decode(ind)
    targets = list(g.edges) + [n for n in g.nodes if n not in ("In", "Out")]
    if not targets:
        return ind
    for _ in range(DELETE_MUTATION_TRIES):
        target = random.choice(targets)
        if isinstance(target, tuple):
            success, new_g = g.delete_edge_valid(target)
        else:
            success, new_g = g.delete_node_valid(target)
        if success:
            return encode(new_g)
    return ind


MUTATION_FNS = [
    mutate_counts,
    mutate_counts_preserve,
    mutate_perm_swap,
    mutate_perm_scramble,
    mutate_perm_reverse,
    mutate_graph_delete,
]
