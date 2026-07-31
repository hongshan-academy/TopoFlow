import random
from collections import defaultdict
from typing import Dict, Tuple, List, Set

from graph import Graph, Node, Edge
from config import DEFAULT_CONFIG as _cfg


def is_balanced(s2: int, s3: int, c2: int, c3: int) -> bool:
    return s2 + 2 * s3 == c2 + 2 * c3


def port_count(s2: int, s3: int, c2: int, c3: int) -> int:
    return 1 + 2 * s2 + 3 * s3 + c2 + c3


def _make_out_ports(s2: int, s3: int, c2: int, c3: int) -> List[Tuple[Node, int]]:
    out_ports: List[Tuple[Node, int]] = [("In", 0)]
    for i in range(s2):
        out_ports.append((f"S2_{i}", 0))
        out_ports.append((f"S2_{i}", 1))
    for i in range(s3):
        out_ports.append((f"S3_{i}", 0))
        out_ports.append((f"S3_{i}", 1))
        out_ports.append((f"S3_{i}", 2))
    for i in range(c2):
        out_ports.append((f"C2_{i}", 0))
    for i in range(c3):
        out_ports.append((f"C3_{i}", 0))
    return out_ports


def _make_in_ports(s2: int, s3: int, c2: int, c3: int) -> List[Tuple[Node, int]]:
    in_ports: List[Tuple[Node, int]] = []
    for i in range(s2):
        in_ports.append((f"S2_{i}", 0))
    for i in range(s3):
        in_ports.append((f"S3_{i}", 0))
    for i in range(c2):
        in_ports.append((f"C2_{i}", 0))
        in_ports.append((f"C2_{i}", 1))
    for i in range(c3):
        in_ports.append((f"C3_{i}", 0))
        in_ports.append((f"C3_{i}", 1))
        in_ports.append((f"C3_{i}", 2))
    in_ports.append(("Out", 0))
    return in_ports


def repair_perm(perm: List[int], out_ports: List[Tuple[Node, int]], in_ports: List[Tuple[Node, int]]) -> List[int]:
    n = len(perm)
    perm = list(perm)
    for i in range(n):
        if not (0 <= perm[i] < len(in_ports)):
            perm = list(range(n))
            random.shuffle(perm)
            break

    changed = True
    iterations = 0
    max_iter = n * _cfg.repair_max_iter_factor

    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        for i in range(n):
            if out_ports[i][0] != in_ports[perm[i]][0]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if out_ports[i][0] != in_ports[perm[j]][0] and out_ports[j][0] != in_ports[perm[i]][0]:
                    perm[i], perm[j] = perm[j], perm[i]
                    changed = True
                    break

    if _cfg.repair_retry_on_fail:
        for i in range(n):
            if out_ports[i][0] == in_ports[perm[i]][0]:
                perm = list(range(n))
                random.shuffle(perm)
                break

    return perm


def balance_counts(s2: int, s3: int, c2: int, c3: int, _depth: int = 0) -> Tuple[int, int, int, int]:
    s2, s3, c2, c3 = max(0, s2), max(0, s3), max(0, c2), max(0, c3)
    left = s2 + 2 * s3
    right = c2 + 2 * c3

    if left == right:
        return s2, s3, c2, c3

    if _depth > _cfg.balance_max_depth:
        if left > right:
            c2, c3 = left, 0
        else:
            s2, s3 = right, 0
        return max(0, s2), max(0, s3), max(0, c2), max(0, c3)

    orig = (s2, s3, c2, c3)

    if left > right:
        deficit = left - right
        while deficit >= 2 and s3 > 0:
            s3 -= 1
            deficit -= 2
        while deficit >= 1 and s2 > 0:
            s2 -= 1
            deficit -= 1
        while deficit > 0:
            c2 += 1
            deficit -= 1
    else:
        surplus = right - left
        while surplus >= 2 and c3 > 0:
            c3 -= 1
            surplus -= 2
        while surplus >= 1 and c2 > 0:
            c2 -= 1
            surplus -= 1
        while surplus > 0:
            s2 += 1
            surplus -= 1

    if (s2, s3, c2, c3) == orig:
        return 0, 0, 0, 0

    return balance_counts(s2, s3, c2, c3, _depth + 1)


def make_random_chromosome(n_internal: int) -> Tuple[int, ...]:
    for _ in range(_cfg.chromosome_retries):
        s2 = random.randint(0, n_internal)
        s3 = random.randint(0, n_internal - s2)
        remain = n_internal - s2 - s3
        c3 = s2 + 2 * s3 - remain
        c2 = remain - c3
        if c3 >= 0 and c2 >= 0:
            s2, s3, c2, c3 = balance_counts(s2, s3, c2, c3)
            n = port_count(s2, s3, c2, c3)
            out_ports = _make_out_ports(s2, s3, c2, c3)
            in_ports = _make_in_ports(s2, s3, c2, c3)
            perm = list(range(n))
            random.shuffle(perm)
            perm = repair_perm(perm, out_ports, in_ports)
            return (s2, s3, c2, c3) + tuple(perm)

    s2, s3, c2, c3 = 2, 0, 2, 0
    n = port_count(s2, s3, c2, c3)
    out_ports = _make_out_ports(s2, s3, c2, c3)
    in_ports = _make_in_ports(s2, s3, c2, c3)
    perm = list(range(n))
    random.shuffle(perm)
    perm = repair_perm(perm, out_ports, in_ports)
    return (s2, s3, c2, c3) + tuple(perm)


def get_counts(ind: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    return ind[0], ind[1], ind[2], ind[3]


def get_perm(ind: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(ind[4:])


def decode(ind: Tuple[int, ...]) -> Graph:
    s2, s3, c2, c3 = ind[:4]
    perm = ind[4:]
    n = port_count(s2, s3, c2, c3)
    assert len(perm) == n, f"perm length {len(perm)} != port count {n}"

    out_ports = _make_out_ports(s2, s3, c2, c3)
    in_ports = _make_in_ports(s2, s3, c2, c3)
    perm_list: List[int] = repair_perm(list(perm), out_ports, in_ports)
    perm = tuple(perm_list)

    nodes: Set[Node] = {"In", "Out"}
    for i in range(s2):
        nodes.add(f"S2_{i}")
    for i in range(s3):
        nodes.add(f"S3_{i}")
    for i in range(c2):
        nodes.add(f"C2_{i}")
    for i in range(c3):
        nodes.add(f"C3_{i}")

    edges: List[Edge] = []
    for out_idx, in_idx in enumerate(perm):
        out_node, _ = out_ports[out_idx]
        in_node, _ = in_ports[in_idx]
        if out_node != in_node:
            edges.append((out_node, in_node))

    return Graph(nodes, edges, _validate=False)


def encode(graph: Graph) -> Tuple[int, ...]:
    s2_count = 0
    s3_count = 0
    c2_count = 0
    c3_count = 0

    for node in graph.nodes:
        if node == "In" or node == "Out":
            continue
        if node.startswith("S2_"):
            s2_count += 1
        elif node.startswith("S3_"):
            s3_count += 1
        elif node.startswith("C2_"):
            c2_count += 1
        elif node.startswith("C3_"):
            c3_count += 1

    name_map: Dict[Node, Node] = {"In": "In", "Out": "Out"}
    new_idx: Dict[str, int] = {"S2": 0, "S3": 0, "C2": 0, "C3": 0}

    for node in graph.nodes:
        if node in ("In", "Out"):
            continue
        for prefix in ("S2", "S3", "C2", "C3"):
            if node.startswith(prefix + "_"):
                name_map[node] = f"{prefix}_{new_idx[prefix]}"
                new_idx[prefix] += 1
                break

    out_ports = _make_out_ports(s2_count, s3_count, c2_count, c3_count)
    in_ports = _make_in_ports(s2_count, s3_count, c2_count, c3_count)

    available_in: Dict[Node, List[int]] = defaultdict(list)
    for i, (node, _) in enumerate(in_ports):
        available_in[node].append(i)

    renamed_edges: Dict[Node, List[Node]] = defaultdict(list)
    for u, v in graph.edges:
        ru = name_map[u]
        rv = name_map[v]
        renamed_edges[ru].append(rv)

    n = len(out_ports)
    perm = [-1] * n

    for i, (out_node, _) in enumerate(out_ports):
        targets = renamed_edges.get(out_node, [])
        matched = False
        for target in targets:
            if target != out_node and target in available_in and available_in[target]:
                perm[i] = available_in[target].pop()
                matched = True
                break
        if not matched:
            for target in targets:
                if target in available_in and available_in[target]:
                    perm[i] = available_in[target].pop()
                    matched = True
                    break
        if not matched:
            for node, ports in available_in.items():
                if node != out_node and ports:
                    perm[i] = ports.pop()
                    matched = True
                    break
        if not matched:
            for node, ports in available_in.items():
                if ports:
                    perm[i] = ports.pop()
                    break

    return (s2_count, s3_count, c2_count, c3_count) + tuple(perm)
