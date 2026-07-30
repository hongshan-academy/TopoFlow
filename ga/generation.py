import random
from typing import Tuple, Optional, List, Set, Dict

from graph import Graph, Node, Edge
from config import DEFAULT_CONFIG as _cfg

_module_counter = 0


def _reachable_from(start: Node, out_edges: Dict[Node, List[Tuple[Node, Node]]]) -> Set[Node]:
    visited: Set[Node] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for _, v in out_edges.get(node, []):
            if v not in visited:
                stack.append(v)
    return visited


def _reachable_to(target: Node, in_edges: Dict[Node, List[Tuple[Node, Node]]]) -> Set[Node]:
    visited: Set[Node] = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for u, _ in in_edges.get(node, []):
            if u not in visited:
                stack.append(u)
    return visited


def _get_balanced_config(total: int) -> Optional[Tuple[int, int, int, int]]:
    for _ in range(_cfg.balanced_config_tries):
        s2 = random.randint(0, total)
        s3 = random.randint(0, total - s2)
        remain = total - s2 - s3
        c3 = s2 + 2 * s3 - remain
        c2 = remain - c3
        if c3 >= 0 and c2 >= 0:
            return s2, s3, c2, c3
    return None


def _generate_wire_graph(s2: int, s3: int, c2: int, c3: int) -> Optional[Graph]:
    output_wires: List[Tuple[str, int]] = [("In", 0)]
    input_wires: List[Tuple[str, int]] = [("Out", 0)]

    s2_names = [f"S2_{i}" for i in range(s2)]
    s3_names = [f"S3_{i}" for i in range(s3)]
    c2_names = [f"C2_{i}" for i in range(c2)]
    c3_names = [f"C3_{i}" for i in range(c3)]

    for name in s2_names:
        input_wires.append((name, 0))
        output_wires.extend([(name, 0), (name, 1)])
    for name in s3_names:
        input_wires.append((name, 0))
        output_wires.extend([(name, 0), (name, 1), (name, 2)])
    for name in c2_names:
        input_wires.extend([(name, 0), (name, 1)])
        output_wires.append((name, 0))
    for name in c3_names:
        input_wires.extend([(name, 0), (name, 1), (name, 2)])
        output_wires.append((name, 0))

    all_nodes = {"In", "Out"} | set(s2_names + s3_names + c2_names + c3_names)

    for _ in range(_cfg.max_generation_tries):
        shuffled = output_wires.copy()
        random.shuffle(shuffled)

        edges: List[Tuple[str, str]] = []
        bad = False
        seen_pairs: Set[Tuple[str, str]] = set()

        for (out_n, _), (in_n, _) in zip(shuffled, input_wires):
            if out_n == in_n:
                bad = True
                break
            if (out_n, in_n) in seen_pairs:
                bad = True
                break
            seen_pairs.add((out_n, in_n))
            edges.append((out_n, in_n))

        if bad:
            continue

        graph = Graph(all_nodes, edges, _validate=False)

        if not graph.is_valid(strict=True):
            continue
        if "In" not in graph.sources or "Out" not in graph.sinks:
            continue

        fwd = _reachable_from("In", graph.out_edges)
        bwd = _reachable_to("Out", graph.in_edges)

        all_connected = True
        for node in all_nodes:
            if node in ("In", "Out"):
                continue
            if node not in fwd or node not in bwd:
                all_connected = False
                break

        if all_connected:
            return graph

    return None


def generate_strict_graph(
    n_internal: int,
    source: Node = "In",
    sink: Node = "Out",
) -> Graph:
    if n_internal < 2:
        n_internal = 2

    cfg = _get_balanced_config(n_internal)
    if cfg is None:
        return Graph({source, sink}, [(source, sink)], _validate=False)

    s2, s3, c2, c3 = cfg
    if s2 + s3 == 0 or c2 + c3 == 0:
        return Graph({source, sink}, [(source, sink)], _validate=False)

    graph = _generate_wire_graph(s2, s3, c2, c3)
    if graph is None:
        return Graph({source, sink}, [(source, sink)], _validate=False)

    return graph


def generate_subgraph(
    n_internal: int,
) -> Tuple[Graph, Node, Node]:
    min_n = max(2, n_internal)
    cfg = _get_balanced_config(min_n)
    if cfg is None:
        entry = "E"
        exit_node = "X"
        return (
            Graph({entry, exit_node}, [(entry, exit_node), (entry, exit_node)], _validate=False),
            entry,
            exit_node,
        )

    s2, s3, c2, c3 = cfg
    if s2 + s3 == 0 or c2 + c3 == 0:
        entry = "E"
        exit_node = "X"
        return (
            Graph({entry, exit_node}, [(entry, exit_node), (entry, exit_node)], _validate=False),
            entry,
            exit_node,
        )

    output_wires: List[Tuple[str, int]] = [("E", 0), ("E", 1)]
    input_wires: List[Tuple[str, int]] = [("X", 0), ("X", 1)]

    s2_names = [f"FS2_{i}" for i in range(s2)]
    s3_names = [f"FS3_{i}" for i in range(s3)]
    c2_names = [f"FC2_{i}" for i in range(c2)]
    c3_names = [f"FC3_{i}" for i in range(c3)]

    for name in s2_names:
        input_wires.append((name, 0))
        output_wires.extend([(name, 0), (name, 1)])
    for name in s3_names:
        input_wires.append((name, 0))
        output_wires.extend([(name, 0), (name, 1), (name, 2)])
    for name in c2_names:
        input_wires.extend([(name, 0), (name, 1)])
        output_wires.append((name, 0))
    for name in c3_names:
        input_wires.extend([(name, 0), (name, 1), (name, 2)])
        output_wires.append((name, 0))

    all_nodes = {"E", "X"} | set(s2_names + s3_names + c2_names + c3_names)

    for _ in range(_cfg.max_generation_tries):
        shuffled = output_wires.copy()
        random.shuffle(shuffled)

        edges: List[Tuple[str, str]] = []
        bad = False
        seen_pairs: Set[Tuple[str, str]] = set()

        for (out_n, _), (in_n, _) in zip(shuffled, input_wires):
            if out_n == in_n:
                bad = True
                break
            if (out_n, in_n) in seen_pairs:
                bad = True
                break
            seen_pairs.add((out_n, in_n))
            edges.append((out_n, in_n))

        if bad:
            continue

        graph = Graph(all_nodes, edges, _validate=False)

        entry_deg = tuple(graph.degrees.get("E", [0, 0]))
        exit_deg = tuple(graph.degrees.get("X", [0, 0]))
        if entry_deg[0] != 0 or entry_deg[1] == 0:
            continue
        if exit_deg[1] != 0 or exit_deg[0] == 0:
            continue

        fwd = _reachable_from("E", graph.out_edges)
        bwd = _reachable_to("X", graph.in_edges)

        all_connected = True
        for node in all_nodes:
            if node in ("E", "X"):
                continue
            if node not in fwd or node not in bwd:
                all_connected = False
                break

        if all_connected:
            return graph, "E", "X"

    entry = "E"
    exit_node = "X"
    return (
        Graph({entry, exit_node}, [(entry, exit_node), (entry, exit_node)], _validate=False),
        entry,
        exit_node,
    )


def generate_replacement_module(
    n_internal: int,
) -> Optional[Tuple[Graph, Node, Node]]:
    """生成完整图 → 切掉 In/Out → 返回 (subgraph, entry, exit).

    entry 节点切掉 In 后欠 1 条入边，exit 节点切掉 Out 后欠 1 条出边。
    插入父图时补上 u→entry 和 exit→v，度数完美还原。
    返回 None 表示生成退化图（仅 In→Out 无内部结构）。
    """
    global _module_counter
    _module_counter += 1
    tag = f"__{_module_counter}"

    full = generate_strict_graph(n_internal)

    if len(full.edges) == 1:
        return None

    source_out = full.out_edges.get("In", [])
    sink_in = full.in_edges.get("Out", [])
    if not source_out or not sink_in:
        return None

    _, entry_old = source_out[0]
    exit_old, _ = sink_in[0]

    rename: Dict[Node, Node] = {}
    for node in full.nodes:
        if node not in ("In", "Out"):
            rename[node] = node + tag

    edges: List[Edge] = []
    for u, v in full.edges:
        if u == "In" or u == "Out" or v == "In" or v == "Out":
            continue
        edges.append((rename.get(u, u), rename.get(v, v)))

    nodes_set: Set[Node] = set()
    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)

    if not nodes_set:
        return None

    subgraph = Graph(nodes_set, edges, _validate=False)
    return subgraph, rename.get(entry_old, entry_old), rename.get(exit_old, exit_old)
