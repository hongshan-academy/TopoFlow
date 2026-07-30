from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from result import EdgeResult, SimulatorResult
from graph import Graph, Node, Edge, NodeType


EDGE_CAPACITY = 2

EdgeKey = Tuple[Node, Node, int]


@dataclass
class NodeRuntime:
    has_item: bool = False
    rr_in_index: int = -1
    rr_out_index: int = -1


class TopoFlowSimulator:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.node_types: Dict[Node, NodeType] = {
            node: graph.classify(node) for node in graph.nodes
        }

        self._all_edges: List[EdgeKey] = []
        self._out_slots: Dict[Node, List[EdgeKey]] = {node: [] for node in graph.nodes}
        self._in_slots: Dict[Node, List[EdgeKey]] = {node: [] for node in graph.nodes}
        for i, (u, v) in enumerate(graph.edges):
            key: EdgeKey = (u, v, i)
            self._all_edges.append(key)
            self._out_slots[u].append(key)
            self._in_slots[v].append(key)

        self._out_slot_index: Dict[Node, Dict[EdgeKey, int]] = {
            node: {e: i for i, e in enumerate(self._out_slots[node])}
            for node in graph.nodes
        }
        self._in_slot_index: Dict[Node, Dict[EdgeKey, int]] = {
            node: {e: i for i, e in enumerate(self._in_slots[node])}
            for node in graph.nodes
        }

        self.update_order = self._compute_update_order()
        self.deliverable = self._compute_deliverable()

        self.frame = 0
        self.global_cycle_info: Optional[Dict[str, Any]] = None

        self.node_runtime: Dict[Node, NodeRuntime] = {}
        self.edge_queues: Dict[EdgeKey, Deque[int]] = {}

        self.reset()

    def reset(self) -> None:
        self.frame = 0
        self.node_runtime.clear()
        self.edge_queues.clear()
        self.global_cycle_info = None

        for node in self.graph.nodes:
            self.node_runtime[node] = NodeRuntime()

        for node in self.graph.nodes:
            for edge in self._out_slots[node]:
                self.edge_queues.setdefault(edge, deque(maxlen=EDGE_CAPACITY))

    def _compute_update_order(self) -> List[Node]:
        visited: Set[Node] = set()
        order: List[Node] = []
        queue: Deque[Node] = deque()

        def push(n: Node) -> None:
            if n not in visited:
                visited.add(n)
                queue.append(n)

        for node in self.graph.nodes:
            if self.node_types[node] == NodeType.SOURCE:
                push(node)

        while queue:
            node = queue.popleft()
            order.append(node)
            for _, target in self.graph.out_edges[node]:
                push(target)

        for node in self.graph.nodes:
            if node in visited:
                continue
            push(node)
            while queue:
                current = queue.popleft()
                order.append(current)
                for _, target in self.graph.out_edges[current]:
                    push(target)

        return order

    def _compute_deliverable(self) -> Set[Node]:
        deliverable: Set[Node] = set()
        queue: Deque[Node] = deque()

        for node in self.graph.nodes:
            if self.node_types[node] == NodeType.SINK:
                deliverable.add(node)
                queue.append(node)

        while queue:
            node = queue.popleft()
            for source, _ in self.graph.in_edges[node]:
                if source not in deliverable:
                    deliverable.add(source)
                    queue.append(source)

        return deliverable

    def _can_edge_deliver(self, edge: EdgeKey) -> bool:
        return edge[1] in self.deliverable

    @staticmethod
    def _pick_round_robin(
        slot_count: int, current: int, has_candidate: Any
    ) -> int:
        if slot_count <= 0:
            return -1
        for offset in range(1, slot_count + 1):
            idx = (current + offset) % slot_count
            if has_candidate(idx):
                return idx
        return -1

    def _choose_outgoing_edge(
        self,
        node: Node,
        frame: List[int],
        edge_state: Dict[EdgeKey, Deque[int]],
    ) -> Optional[EdgeKey]:
        node_type = self.node_types[node]
        out_edges = self._out_slots[node]

        if node_type == NodeType.SINK or not out_edges:
            return None

        if node_type == NodeType.SPLITTER:
            idx = self._pick_round_robin(
                len(out_edges),
                frame[2],
                lambda i: (
                    len(edge_state[out_edges[i]]) < EDGE_CAPACITY
                    and self._can_edge_deliver(out_edges[i])
                ),
            )
            return out_edges[idx] if idx != -1 else None

        edge = out_edges[0]
        if len(edge_state[edge]) >= EDGE_CAPACITY or not self._can_edge_deliver(edge):
            return None
        return edge

    def _choose_incoming_edge(
        self,
        node: Node,
        frame: List[int],
        edge_start_items: Dict[EdgeKey, int],
        edge_state: Dict[EdgeKey, Deque[int]],
    ) -> Optional[EdgeKey]:
        node_type = self.node_types[node]
        in_edges = self._in_slots[node]

        if node_type == NodeType.SOURCE or not in_edges:
            return None

        available = [
            e for e in in_edges
            if edge_start_items[e] > 0 and len(edge_state[e]) > 0
        ]

        if not available:
            return None

        if node_type == NodeType.CONVERGER:
            available_slots = {self._in_slot_index[node][e] for e in available}
            idx = self._pick_round_robin(
                len(in_edges),
                frame[1],
                lambda i: i in available_slots,
            )
            if idx == -1:
                return None
            for e in available:
                if self._in_slot_index[node][e] == idx:
                    return e
            return None

        return available[0]

    def _node_has_receive_space(self, node: Node, frame: List[int]) -> bool:
        node_type = self.node_types[node]
        if node_type == NodeType.SOURCE:
            return False
        if node_type == NodeType.SINK:
            return True
        if not self._out_slots[node]:
            return False
        return not frame[0]

    def step_once(self) -> Tuple[Dict[Node, bool], Set[EdgeKey]]:
        node_start: Dict[Node, Tuple[bool, int, int]] = {}
        node_state: Dict[Node, List[int]] = {}
        node_sent: Dict[Node, bool] = {}
        node_received: Dict[Node, bool] = {}
        edge_start_items: Dict[EdgeKey, int] = {}
        edge_state: Dict[EdgeKey, Deque[int]] = {}
        edge_filled: Set[EdgeKey] = set()

        queue: Deque[Node] = deque()
        in_queue: Set[Node] = set()

        for node in self.graph.nodes:
            rt = self.node_runtime[node]
            snap = (rt.has_item, rt.rr_in_index, rt.rr_out_index)
            node_start[node] = snap
            node_state[node] = list(snap)
            node_sent[node] = False
            node_received[node] = False

        for edge in self._all_edges:
            q = self.edge_queues[edge]
            edge_start_items[edge] = len(q)
            edge_state[edge] = deque(q, maxlen=EDGE_CAPACITY)

        def enqueue(n: Node) -> None:
            if n not in in_queue:
                in_queue.add(n)
                queue.append(n)

        for node in reversed(self.update_order):
            enqueue(node)

        while queue:
            node = queue.popleft()
            in_queue.discard(node)

            rt = self.node_runtime[node]
            current = node_state[node]
            node_type = self.node_types[node]

            started_with_supply = (
                True if node_type == NodeType.SOURCE else node_start[node][0]
            )

            if started_with_supply and not node_sent[node]:
                send_edge = self._choose_outgoing_edge(node, current, edge_state)
                if send_edge is not None:
                    edge_state[send_edge].append(1)
                    edge_filled.add(send_edge)
                    node_sent[node] = True
                    if node_type != NodeType.SOURCE:
                        current[0] = False
                    if node_type == NodeType.SPLITTER:
                        current[2] = self._out_slot_index[node][send_edge]

            if not node_received[node] and self._node_has_receive_space(node, current):
                incoming = self._choose_incoming_edge(
                    node, current, edge_start_items, edge_state
                )
                if incoming is not None:
                    edge_state[incoming].popleft()
                    edge_start_items[incoming] -= 1
                    node_received[node] = True
                    if node_type != NodeType.SINK:
                        current[0] = True
                    if node_type == NodeType.CONVERGER:
                        current[1] = self._in_slot_index[node][incoming]
                    enqueue(incoming[0])

        for node in self.graph.nodes:
            rt = self.node_runtime[node]
            current = node_state[node]
            node_type = self.node_types[node]
            if node_type not in (NodeType.SOURCE, NodeType.SINK):
                rt.has_item = bool(current[0])
            rt.rr_in_index = current[1]
            rt.rr_out_index = current[2]

        for edge in self._all_edges:
            self.edge_queues[edge] = edge_state[edge]

        self.frame += 1
        return node_received, edge_filled

    def serialize_state(self) -> Tuple[int, ...]:
        node_part: List[int] = []
        for node in self.update_order:
            rt = self.node_runtime[node]
            node_part.extend((1 if rt.has_item else 0, rt.rr_in_index, rt.rr_out_index))
        edge_part: List[int] = []
        for edge in self._all_edges:
            edge_part.extend(self.edge_queues[edge])
        return tuple(node_part + edge_part)

    def run_until_cycle(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        state_to_frame: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        node_flow_ones: Dict[Node, int] = {
            node: 0 for node in self.graph.nodes
        }
        edge_flow_ones: Dict[EdgeKey, int] = {
            edge: 0 for edge in self._all_edges
        }

        while True:
            if max_frames is not None and self.frame >= max_frames:
                self.global_cycle_info = _build_max_frames_result(
                    self, node_flow_ones, edge_flow_ones,
                )
                return self.global_cycle_info

            key = self.serialize_state()

            if key in state_to_frame:
                stored = state_to_frame[key]
                self.global_cycle_info = _build_cycle_result(
                    self, stored, node_flow_ones, edge_flow_ones,
                )
                return self.global_cycle_info

            state_to_frame[key] = {
                'frame': self.frame,
                'node_flow_ones': dict(node_flow_ones),
                'edge_flow_ones': dict(edge_flow_ones),
            }

            node_received, edge_filled = self.step_once()

            for node, received in node_received.items():
                if received:
                    node_flow_ones[node] += 1

            for edge in edge_filled:
                edge_flow_ones[edge] += 1

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            'frame': self.frame,
            'nodes': [
                {
                    'id': node,
                    'type': self.node_types[node].name,
                    'has_item': self.node_runtime[node].has_item,
                    'rr_in_index': self.node_runtime[node].rr_in_index,
                    'rr_out_index': self.node_runtime[node].rr_out_index,
                }
                for node in self.graph.nodes
            ],
            'edges': [
                {
                    'id': f'{e[0]}->{e[1]}',
                    'from': e[0],
                    'to': e[1],
                    'idx': e[2],
                    'queue': list(self.edge_queues[e]),
                }
                for e in self._all_edges
            ],
        }


def _build_cycle_result(
    sim: TopoFlowSimulator,
    stored: Dict[str, Any],
    node_flow_ones: Dict[Node, int],
    edge_flow_ones: Dict[EdgeKey, int],
    converged: bool = True,
) -> Dict[str, Any]:
    period = sim.frame - stored['frame']
    node_ratios: Dict[Node, Tuple[int, int]] = {}
    for node in sim.graph.nodes:
        num = node_flow_ones[node] - stored['node_flow_ones'][node]
        node_ratios[node] = (num, period)
    edge_ratios: Dict[EdgeKey, Tuple[int, int]] = {}
    for edge in sim._all_edges:
        num = edge_flow_ones[edge] - stored['edge_flow_ones'][edge]
        edge_ratios[edge] = (num, period)
    return {
        'period': period,
        'cycle_start_frame': stored['frame'],
        'warmup_frames': stored['frame'],
        'total_frames': sim.frame,
        'node_ratios': node_ratios,
        'edge_ratios': edge_ratios,
        'converged': converged,
    }


def _build_max_frames_result(
    sim: TopoFlowSimulator,
    node_flow_ones: Dict[Node, int],
    edge_flow_ones: Dict[EdgeKey, int],
) -> Dict[str, Any]:
    return {
        'period': 1,
        'cycle_start_frame': 0,
        'warmup_frames': sim.frame,
        'total_frames': sim.frame,
        'node_ratios': {n: (node_flow_ones[n], 1) for n in sim.graph.nodes},
        'edge_ratios': {e: (edge_flow_ones[e], 1) for e in sim._all_edges},
        'converged': False,
    }


def simulate(graph: Graph, max_frames: Optional[int] = None) -> SimulatorResult:
    sim = TopoFlowSimulator(graph)
    cycle = sim.run_until_cycle(max_frames=max_frames)

    edge_results: List[EdgeResult] = []
    for edge in sim._all_edges:
        num, den = cycle['edge_ratios'][edge]
        flow = num / den if den > 0 else 0.0
        edge_results.append(EdgeResult(
            source=edge[0],
            target=edge[1],
            flow=flow,
            is_blocked=False,
            is_full=False,
        ))

    return SimulatorResult(edges=edge_results, converged=cycle.get('converged', True))


def simulate_frames(graph: Graph, max_frames: Optional[int] = None) -> Dict[str, Any]:
    sim = TopoFlowSimulator(graph)

    frames: List[Dict[str, Any]] = [sim.get_snapshot()]
    state_to_frame: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    node_flow_ones: Dict[Node, int] = {node: 0 for node in graph.nodes}
    edge_flow_ones: Dict[EdgeKey, int] = {edge: 0 for edge in sim._all_edges}

    while True:
        if max_frames is not None and sim.frame >= max_frames:
            return {
                'cycle': _build_max_frames_result(sim, node_flow_ones, edge_flow_ones),
                'frames': frames,
            }

        key = sim.serialize_state()

        if key in state_to_frame:
            stored = state_to_frame[key]
            return {
                'cycle': _build_cycle_result(sim, stored, node_flow_ones, edge_flow_ones),
                'frames': frames,
            }

        state_to_frame[key] = {
            'frame': sim.frame,
            'node_flow_ones': dict(node_flow_ones),
            'edge_flow_ones': dict(edge_flow_ones),
        }

        node_received, edge_filled = sim.step_once()

        for node, received in node_received.items():
            if received:
                node_flow_ones[node] += 1

        for edge in edge_filled:
            edge_flow_ones[edge] += 1

        frames.append(sim.get_snapshot())
