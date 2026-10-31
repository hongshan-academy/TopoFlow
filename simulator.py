"""Tick-level TopoFlow simulator with a frame-replay adapter.

The physical model follows the sibling ``topo-layout`` engine:

* one tick is 0.25 s and one replay frame is ``8`` ticks;
* a belt cell has eight positions and an item needs at least eight ticks to
  cross one cell (``8 * length`` ticks for a belt of ``length`` cells);
* a pipe cell has two positions and an item needs at least two ticks to cross
  one cell (``2 * length`` ticks for a pipe of ``length`` cells);
* CU/ICU/PCU admission uses an arrival-anchored forty-tick window (five
  replay frames) that accepts at most ``gear`` items.

The simulator keeps the complete sub-frame state: every item's position on
every edge, node buffers, round-robin cursors, source cooldowns and the
control-port window phase.  ``serialize_state`` packs that state into a compact
byte string, and cycle detection compares a 160-bit digest of it at tick
resolution.  Sampling only every eight ticks would alias short periods and
could merge physically different states, so it is deliberately not used.

``TopoFlowSimulator.step_once`` advances one replay frame and is kept for the
existing web API and for callers that treat a frame as the unit of work.
``simulate_frames`` drives the simulator tick by tick, reports the true tick
period, and returns snapshots only at frame boundaries.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graph import Graph, Node, NodeType, control_gear, control_mode


TICKS_PER_FRAME = 8
DEFAULT_EDGE_LENGTH = 4
# Kept for the web UI, which still labels edge occupancy as ``n / 4``.
EDGE_CAPACITY = DEFAULT_EDGE_LENGTH
CONTROL_WINDOW_TICKS = 40
BELT_SPACING = 8
PIPE_SPACING = 2
STATE_DIGEST_BYTES = 20

EdgeKey = Tuple[Node, Node, int]


@dataclass
class NodeRuntime:
    """Mutable per-node state, including the tick-phase fields.

    ``cu_window_start`` is only a presentation convenience (absolute tick);
    ``cu_window_age`` is the state that participates in cycle detection.
    """

    has_item: bool = False
    rr_in_index: int = -1
    rr_out_index: int = -1
    cu_window_age: int = -1
    cu_window_received: int = 0
    cu_window_start: Optional[int] = None
    source_cooldown: int = 0
    source_enabled: bool = True


class _Transport:
    """Sparse shift register: item age in ticks, downstream first.

    ``remaining == 1`` means the item is at the downstream end and may be
    received.  A just-appended item starts at ``maxlen + 1`` (one position
    before the belt), mirroring ``topo-layout``'s ``CellTransport``.
    """

    __slots__ = (
        "length",
        "kind",
        "spacing",
        "maxlen",
        "capacity",
        "entries",
        "pending",
    )

    def __init__(self, length: int, kind: str) -> None:
        if kind not in ("belt", "pipe"):
            raise ValueError(f"unknown transport kind: {kind!r}")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError(f"edge length must be a non-negative integer, got {length!r}")
        self.length = length
        self.kind = kind
        self.spacing = PIPE_SPACING if kind == "pipe" else BELT_SPACING
        self.maxlen = length * self.spacing
        self.capacity = length
        self.entries: Deque[int] = deque()
        self.pending = False

    # -- item movement -------------------------------------------------
    def advance(self) -> None:
        if self.capacity == 0:
            return
        ahead: Optional[int] = None
        for index, remaining in enumerate(self.entries):
            proposed = max(1, remaining - 1)
            if ahead is None or proposed - ahead >= self.spacing:
                self.entries[index] = proposed
            ahead = self.entries[index]

    def can_receive(self) -> bool:
        if self.capacity == 0:
            return not self.pending
        if len(self.entries) >= self.capacity:
            return False
        return (
            not self.entries
            or self.maxlen + 1 - self.entries[-1] >= self.spacing
        )

    def append(self) -> bool:
        """Append at the upstream end (normal producer injection)."""
        if not self.can_receive():
            return False
        if self.capacity == 0:
            self.pending = True
        else:
            self.entries.append(self.maxlen + 1)
        return True

    def inject_at_tail(self) -> bool:
        """Debug hook: make one item immediately available at the tail."""
        if self.capacity == 0:
            if self.pending:
                return False
            self.pending = True
            return True
        if not self.can_receive():
            return False
        if not self.entries:
            self.entries.append(1)
        else:
            # Keep spacing if the tail is already occupied.
            self.entries.append(self.maxlen + 1)
        return True

    def can_send(self) -> bool:
        if self.capacity == 0:
            return self.pending
        return bool(self.entries) and self.entries[0] == 1

    def popleft(self) -> bool:
        if not self.can_send():
            raise IndexError("transport tail is empty")
        if self.capacity == 0:
            self.pending = False
        else:
            self.entries.popleft()
        return True

    def clear(self) -> None:
        self.entries.clear()
        self.pending = False

    def item_count(self) -> int:
        if self.capacity == 0:
            return int(self.pending)
        return len(self.entries)

    def occupancy(self) -> List[bool]:
        """Occupied positions from downstream to upstream (for snapshots)."""
        if self.capacity == 0:
            return [self.pending]
        occupied = [False] * self.maxlen
        for remaining in self.entries:
            position = self.maxlen - remaining
            if 0 <= position < self.maxlen:
                occupied[position] = True
        return occupied

    def positions(self) -> List[int]:
        if self.capacity == 0:
            return [] if not self.pending else [0]
        return [self.maxlen - remaining for remaining in self.entries]

    def state_bits(self) -> Tuple[int, int]:
        """Return ``(mask, byte_count)`` for a compact state key.

        Bit zero represents an item still pending one position before the
        belt; bit ``position + 1`` represents an item at ``position``.
        """
        if self.capacity == 0:
            return (1 if self.pending else 0), 1
        mask = 0
        for remaining in self.entries:
            position = self.maxlen - remaining
            mask |= 1 << (position + 1)
        return mask, (self.maxlen + 8) // 8


class _EdgeQueue:
    """Legacy ``deque``-like view over a transport's item count.

    The old simulator exposed ``edge_queues``; tools and tests use ``len()``,
    ``append(1)`` and ``clear()``.  This view keeps those helpers working while
    the transport remains the source of truth.  ``append`` on an empty edge
    puts the item at the delivery end, which is the useful debug behaviour.
    """

    __slots__ = ("transport",)

    def __init__(self, transport: _Transport) -> None:
        self.transport = transport

    def __len__(self) -> int:
        return self.transport.item_count()

    def __bool__(self) -> bool:
        return bool(len(self))

    def __iter__(self):
        return iter([1] * len(self))

    def __getitem__(self, index: int) -> int:
        items = [1] * len(self)
        return items[index]

    def append(self, _item: int = 1) -> None:
        self.transport.inject_at_tail()

    def clear(self) -> None:
        self.transport.clear()

    def popleft(self) -> int:
        self.transport.popleft()
        return 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_EdgeQueue({list(self)!r})"


class TopoFlowSimulator:
    def __init__(
        self,
        graph: Graph,
        *,
        transport: Optional[str] = None,
        edge_length: Optional[int] = None,
        edge_lengths: Optional[Mapping[Any, int] | Sequence[int] | int] = None,
    ) -> None:
        self.graph = graph
        self.node_types: Dict[Node, NodeType] = {
            node: graph.classify(node) for node in graph.nodes
        }
        self.transport = self._resolve_transport(transport)
        self._validate_control_transport()

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

        self._edge_lengths = self._resolve_edge_lengths(edge_length, edge_lengths)
        self._source_interval = (
            PIPE_SPACING if self.transport == "pipe" else BELT_SPACING
        )

        self.tick = 0
        self.frame = 0
        self.node_runtime: Dict[Node, NodeRuntime] = {}
        self.edge_queues: Dict[EdgeKey, _EdgeQueue] = {}
        self._transports: Dict[EdgeKey, _Transport] = {}

        self.reset()

    # -- configuration -------------------------------------------------
    def _resolve_transport(self, transport: Optional[str]) -> str:
        if transport is None:
            modes = {
                mode
                for node in self.graph.nodes
                if (mode := control_mode(node)) is not None
            }
            if "pipe" in modes and "item" in modes:
                raise ValueError(
                    "graph mixes item and pipe control units; pass an explicit transport"
                )
            return "pipe" if "pipe" in modes else "belt"
        if transport in ("belt", "item"):
            return "belt"
        if transport in ("pipe", "fluid"):
            return "pipe"
        raise ValueError(f"transport must be 'belt' or 'pipe', got {transport!r}")

    def _validate_control_transport(self) -> None:
        expected = "pipe" if self.transport == "pipe" else "item"
        wrong = [
            node
            for node in sorted(self.graph.nodes)
            if (mode := control_mode(node)) is not None and mode != expected
        ]
        if wrong:
            raise ValueError(
                f"{self.transport} transport is incompatible with control unit(s): "
                + ", ".join(wrong)
            )

    def _resolve_edge_lengths(
        self,
        edge_length: Optional[int],
        edge_lengths: Optional[Mapping[Any, int] | Sequence[int] | int],
    ) -> List[int]:
        default = DEFAULT_EDGE_LENGTH if edge_length is None else edge_length
        if isinstance(default, bool) or not isinstance(default, int) or default < 0:
            raise ValueError("edge_length must be a non-negative integer")

        if edge_lengths is None:
            values = [default] * len(self._all_edges)
        elif isinstance(edge_lengths, int) and not isinstance(edge_lengths, bool):
            values = [edge_lengths] * len(self._all_edges)
        elif isinstance(edge_lengths, Mapping):
            values = []
            for index, key in enumerate(self._all_edges):
                if index in edge_lengths:
                    value = edge_lengths[index]
                elif key in edge_lengths:
                    value = edge_lengths[key]
                else:
                    value = default
                values.append(value)
        else:
            values = list(edge_lengths)
            if len(values) != len(self._all_edges):
                raise ValueError(
                    f"edge_lengths has {len(values)} entries for "
                    f"{len(self._all_edges)} edges"
                )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("edge lengths must be non-negative integers")
        return values

    # -- topology helpers ---------------------------------------------
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

    def reset(self) -> None:
        self.tick = 0
        self.frame = 0
        self.node_runtime = {node: NodeRuntime() for node in self.graph.nodes}
        self._transports = {
            edge: _Transport(self._edge_lengths[index], self.transport)
            for index, edge in enumerate(self._all_edges)
        }
        self.edge_queues = {
            edge: _EdgeQueue(self._transports[edge]) for edge in self._all_edges
        }

    # -- legacy selection helpers -------------------------------------
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
        frame: Any = None,
        edges: Any = None,
    ) -> Optional[EdgeKey]:
        out_edges = self._out_slots[node]
        if not out_edges:
            return None
        runtime = self.node_runtime[node]
        if len(out_edges) == 1:
            edge = out_edges[0]
            return edge if self._transports[edge].can_receive() else None
        idx = self._pick_round_robin(
            len(out_edges),
            runtime.rr_out_index,
            lambda i: self._transports[out_edges[i]].can_receive(),
        )
        return out_edges[idx] if idx != -1 else None

    def _choose_incoming_edge(
        self,
        node: Node,
        frame: Any = None,
        edge_start_items: Any = None,
        edge_state: Any = None,
    ) -> Optional[EdgeKey]:
        in_edges = self._in_slots[node]
        if not in_edges:
            return None
        runtime = self.node_runtime[node]
        if len(in_edges) == 1:
            edge = in_edges[0]
            return edge if self._transports[edge].can_send() else None
        if self.node_types[node] == NodeType.CONVERGER:
            idx = self._pick_round_robin(
                len(in_edges),
                runtime.rr_in_index,
                lambda i: self._transports[in_edges[i]].can_send(),
            )
            return in_edges[idx] if idx != -1 else None
        for edge in in_edges:
            if self._transports[edge].can_send():
                return edge
        return None

    def _node_has_receive_space(self, node: Node, frame: Any = None) -> bool:
        node_type = self.node_types[node]
        if node_type == NodeType.SOURCE:
            return False
        if node_type == NodeType.SINK:
            return True
        runtime = self.node_runtime[node]
        if runtime.has_item:
            return False
        gear = control_gear(node)
        if gear is not None:
            if runtime.cu_window_age >= 0 and runtime.cu_window_received >= gear:
                return False
        return True

    def set_source_enabled(self, node: Node, enabled: bool) -> None:
        if self.node_types[node] != NodeType.SOURCE:
            raise ValueError(f"{node!r} is not a source")
        self.node_runtime[node].source_enabled = bool(enabled)

    def inject_edge(self, edge: EdgeKey, count: int = 1, cell: int = 0) -> int:
        """Debug hook: append up to ``count`` items at the upstream end."""
        transport = self._transports[edge]
        if cell:
            raise ValueError("cell injection is not supported by the tick model")
        added = 0
        for _ in range(count):
            if not transport.append():
                break
            added += 1
        return added

    # -- tick core -----------------------------------------------------
    def step_tick(self) -> Tuple[Dict[Node, bool], Set[EdgeKey]]:
        """Advance exactly one tick and return the events that happened."""
        for transport in self._transports.values():
            transport.advance()

        node_received: Dict[Node, bool] = {node: False for node in self.graph.nodes}
        edge_filled: Set[EdgeKey] = set()
        initial_has_item = {
            node: runtime.has_item
            for node, runtime in self.node_runtime.items()
        }
        sent: Set[Node] = set()
        received: Set[Node] = set()

        # A tick is a small fixpoint, not a single send/receive pass.  When a
        # receiver takes the tail item, the freed capacity can be reused by its
        # upstream in the same tick; otherwise a saturated belt loses one slot
        # every time the producer happens to be ready exactly on that tick.
        # Each node still sends at most one old item and receives at most one
        # item, and an item received this tick cannot be sent until the next.
        changed = True
        while changed:
            changed = False

            for node in self.update_order:
                if node in sent:
                    continue
                runtime = self.node_runtime[node]
                node_type = self.node_types[node]
                if node_type == NodeType.SOURCE:
                    if not runtime.source_enabled or runtime.source_cooldown > 0:
                        continue
                    edge = self._choose_outgoing_edge(
                        node, self.frame, self._out_slots[node]
                    )
                    if edge is not None and self._transports[edge].append():
                        edge_filled.add(edge)
                        sent.add(node)
                        runtime.source_cooldown = self._source_interval
                        if len(self._out_slots[node]) > 1:
                            runtime.rr_out_index = self._out_slot_index[node][edge]
                        changed = True
                    continue

                if not initial_has_item[node] or not runtime.has_item:
                    continue
                edge = self._choose_outgoing_edge(
                    node, self.frame, self._out_slots[node]
                )
                if edge is None or not self._transports[edge].append():
                    continue
                runtime.has_item = False
                sent.add(node)
                edge_filled.add(edge)
                if len(self._out_slots[node]) > 1:
                    runtime.rr_out_index = self._out_slot_index[node][edge]
                changed = True

            for node in self.update_order:
                if node in received:
                    continue
                node_type = self.node_types[node]
                if node_type == NodeType.SOURCE:
                    continue
                if not self._node_has_receive_space(node):
                    continue
                edge = self._choose_incoming_edge(
                    node, self.frame, self._in_slots[node], self._transports
                )
                if edge is None or not self._transports[edge].can_send():
                    continue
                self._transports[edge].popleft()
                runtime = self.node_runtime[node]
                received.add(node)
                node_received[node] = True
                changed = True
                if node_type != NodeType.SINK:
                    runtime.has_item = True
                if len(self._in_slots[node]) > 1:
                    runtime.rr_in_index = self._in_slot_index[node][edge]
                gear = control_gear(node)
                if gear is not None:
                    if runtime.cu_window_age < 0:
                        runtime.cu_window_age = 0
                        runtime.cu_window_received = 0
                        runtime.cu_window_start = self.tick
                    runtime.cu_window_received += 1

        # Advance control windows after any receipt in this tick.
        for node in self.graph.nodes:
            if control_gear(node) is None:
                continue
            runtime = self.node_runtime[node]
            if runtime.cu_window_age >= 0:
                runtime.cu_window_age += 1
                if runtime.cu_window_age >= CONTROL_WINDOW_TICKS:
                    runtime.cu_window_age = -1
                    runtime.cu_window_received = 0
                    runtime.cu_window_start = None

        for node in self.graph.nodes:
            if self.node_types[node] != NodeType.SOURCE:
                continue
            runtime = self.node_runtime[node]
            if runtime.source_cooldown > 0:
                runtime.source_cooldown -= 1

        self.tick += 1
        if self.tick % TICKS_PER_FRAME == 0:
            self.frame += 1
        return node_received, edge_filled

    def step_once(self) -> Tuple[Dict[Node, bool], Set[EdgeKey]]:
        """Advance one eight-tick replay frame."""
        node_received: Dict[Node, bool] = {node: False for node in self.graph.nodes}
        edge_filled: Set[EdgeKey] = set()
        target = self.tick + TICKS_PER_FRAME
        while self.tick < target:
            received, filled = self.step_tick()
            for node, value in received.items():
                if value:
                    node_received[node] = True
            edge_filled |= filled
        return node_received, edge_filled

    # -- serialization -------------------------------------------------
    def serialize_state(self) -> bytes:
        """Pack the complete physical state into a compact byte string.

        Node fields are packed into four bytes each; an edge is a bitmask with
        one bit per position (plus one pending bit).  Only physical state is
        included: monotonic counters and absolute ticks are excluded.
        """
        buffer = bytearray()
        for node in self.update_order:
            runtime = self.node_runtime[node]
            packed = (
                (1 if runtime.has_item else 0)
                | ((runtime.rr_in_index + 1) & 0xF) << 1
                | ((runtime.rr_out_index + 1) & 0xF) << 5
                | ((runtime.cu_window_age + 1) & 0x3F) << 9
                | (runtime.cu_window_received & 0xF) << 15
                | (runtime.source_cooldown & 0xF) << 19
                | ((1 if runtime.source_enabled else 0) << 23)
            )
            buffer += packed.to_bytes(4, "little")
        for edge in self._all_edges:
            mask, byte_count = self._transports[edge].state_bits()
            buffer += mask.to_bytes(byte_count, "little")
        return bytes(buffer)

    @staticmethod
    def _state_digest(state: bytes) -> bytes:
        return hashlib.blake2b(state, digest_size=STATE_DIGEST_BYTES).digest()

    # -- snapshots -----------------------------------------------------
    def get_snapshot(self, *, include_positions: bool = False) -> Dict[str, Any]:
        """Return a frame-boundary snapshot.

        ``positions`` is opt-in because the replay list stores one snapshot per
        frame; the sub-cell detail is needed by cycle detection, not by the
        default web replay.
        """
        edges = []
        for edge in self._all_edges:
            transport = self._transports[edge]
            entry = {
                "id": f"{edge[0]}->{edge[1]}",
                "from": edge[0],
                "to": edge[1],
                "idx": edge[2],
                "length": transport.length,
                "queue": [1] * transport.item_count(),
            }
            if include_positions:
                entry["positions"] = transport.positions()
            edges.append(entry)
        return {
            "frame": self.frame,
            "tick": self.tick,
            "transport": self.transport,
            "nodes": [
                {
                    "id": node,
                    "type": self.node_types[node].name,
                    "has_item": self.node_runtime[node].has_item,
                    "rr_in_index": self.node_runtime[node].rr_in_index,
                    "rr_out_index": self.node_runtime[node].rr_out_index,
                }
                for node in self.graph.nodes
            ],
            "edges": edges,
        }


# ---------------------------------------------------------------------------
# Cycle driver
# ---------------------------------------------------------------------------

def _count_cycle_events(
    graph: Graph,
    start_tick: int,
    end_tick: int,
    *,
    transport: Optional[str],
    edge_length: Optional[int],
    edge_lengths: Optional[Mapping[Any, int] | Sequence[int] | int],
) -> Tuple[Dict[Node, int], Dict[EdgeKey, int]]:
    """Replay the run and count only the events in ``[start_tick, end_tick)``.

    A state digest table is deliberately not used here: keeping a counter
    snapshot for every tick would dominate memory, while one replay costs only
    a second pass over an already bounded run.
    """
    sim = TopoFlowSimulator(
        graph,
        transport=transport,
        edge_length=edge_length,
        edge_lengths=edge_lengths,
    )
    node_counts: Dict[Node, int] = {node: 0 for node in graph.nodes}
    edge_counts: Dict[EdgeKey, int] = {edge: 0 for edge in sim._all_edges}
    while sim.tick < start_tick:
        sim.step_tick()
    while sim.tick < end_tick:
        received, filled = sim.step_tick()
        for node, value in received.items():
            if value:
                node_counts[node] += 1
        for edge in filled:
            edge_counts[edge] += 1
    return node_counts, edge_counts


def _build_cycle_result(
    *,
    graph: Graph,
    start_tick: int,
    end_tick: int,
    total_tick: int,
    frame_count: int,
    node_counts: Dict[Node, int],
    edge_counts: Dict[EdgeKey, int],
    transport: str,
) -> Dict[str, Any]:
    period_ticks = end_tick - start_tick
    if period_ticks <= 0:
        raise ValueError("cycle period must be positive")
    divisor = math.gcd(period_ticks, TICKS_PER_FRAME)
    period_frames = period_ticks // divisor
    scale = TICKS_PER_FRAME // divisor
    cycle_start_frame = (start_tick + TICKS_PER_FRAME - 1) // TICKS_PER_FRAME

    node_ratios: Dict[Node, Tuple[int, int]] = {
        node: (node_counts[node] * scale, period_frames)
        for node in graph.nodes
    }
    edge_ratios: Dict[EdgeKey, Tuple[int, int]] = {
        edge: (edge_counts[edge] * scale, period_frames)
        for edge in edge_counts
    }
    return {
        "period": period_frames,
        "period_ticks": period_ticks,
        "cycle_start_frame": cycle_start_frame,
        "cycle_start_tick": start_tick,
        "warmup_frames": cycle_start_frame,
        "warmup_ticks": start_tick,
        "total_frames": frame_count,
        "total_ticks": total_tick,
        "node_ratios": node_ratios,
        "edge_ratios": edge_ratios,
        "converged": True,
        "transport": transport,
    }


def _build_max_frames_result(
    sim: TopoFlowSimulator,
    node_flow_ones: Dict[Node, int],
    edge_flow_ones: Dict[EdgeKey, int],
) -> Dict[str, Any]:
    frames = max(sim.frame, 1)
    return {
        "period": 1,
        "period_ticks": TICKS_PER_FRAME,
        "cycle_start_frame": 0,
        "cycle_start_tick": 0,
        "warmup_frames": sim.frame,
        "warmup_ticks": sim.tick,
        "total_frames": sim.frame,
        "total_ticks": sim.tick,
        "node_ratios": {
            node: (node_flow_ones[node], frames) for node in sim.graph.nodes
        },
        "edge_ratios": {
            edge: (edge_flow_ones[edge], frames) for edge in sim._all_edges
        },
        "converged": False,
        "transport": sim.transport,
    }


def simulate_frames(
    graph: Graph,
    max_frames: Optional[int] = None,
    *,
    transport: Optional[str] = None,
    edge_length: Optional[int] = None,
    edge_lengths: Optional[Mapping[Any, int] | Sequence[int] | int] = None,
) -> Dict[str, Any]:
    """Run until the first repeated tick state, returning frame snapshots.

    ``max_frames`` is still a frame budget (one frame = eight ticks) so the
    existing API remains compatible.  The returned cycle carries both the true
    ``period_ticks`` and the frame-compatible ``period``.
    """
    sim = TopoFlowSimulator(
        graph,
        transport=transport,
        edge_length=edge_length,
        edge_lengths=edge_lengths,
    )

    frames: List[Dict[str, Any]] = [sim.get_snapshot()]
    seen: Dict[bytes, int] = {}
    node_flow_ones: Dict[Node, int] = {node: 0 for node in graph.nodes}
    edge_flow_ones: Dict[EdgeKey, int] = {edge: 0 for edge in sim._all_edges}
    max_ticks = None if max_frames is None else max_frames * TICKS_PER_FRAME

    while True:
        if max_ticks is not None and sim.tick >= max_ticks:
            return {
                "cycle": _build_max_frames_result(sim, node_flow_ones, edge_flow_ones),
                "frames": frames,
            }

        state = sim.serialize_state()
        digest = sim._state_digest(state)
        previous = seen.get(digest)
        if previous is not None:
            start_tick = previous
            end_tick = sim.tick
            period_ticks = end_tick - start_tick
            divisor = math.gcd(period_ticks, TICKS_PER_FRAME)
            period_frames = period_ticks // divisor
            cycle_start_frame = (
                start_tick + TICKS_PER_FRAME - 1
            ) // TICKS_PER_FRAME
            target_tick = (cycle_start_frame + period_frames) * TICKS_PER_FRAME
            while sim.tick < target_tick:
                sim.step_tick()
                if sim.tick % TICKS_PER_FRAME == 0:
                    frames.append(sim.get_snapshot())

            node_counts, edge_counts = _count_cycle_events(
                graph,
                start_tick,
                end_tick,
                transport=transport,
                edge_length=edge_length,
                edge_lengths=edge_lengths,
            )
            return {
                "cycle": _build_cycle_result(
                    graph=graph,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    total_tick=sim.tick,
                    frame_count=len(frames) - 1,
                    node_counts=node_counts,
                    edge_counts=edge_counts,
                    transport=sim.transport,
                ),
                "frames": frames,
            }

        seen[digest] = sim.tick
        received, filled = sim.step_tick()
        for node, value in received.items():
            if value:
                node_flow_ones[node] += 1
        for edge in filled:
            edge_flow_ones[edge] += 1

        if sim.tick % TICKS_PER_FRAME == 0:
            frames.append(sim.get_snapshot())
