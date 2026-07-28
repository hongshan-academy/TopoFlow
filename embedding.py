from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from graph import Graph, Node, Edge, NodeType
from config import DEFAULT_CONFIG as _cfg


class PlanarEmbedding:
    graph: Graph
    faces: List[List[Tuple[int, int]]]

    def __init__(self, graph: Graph):
        self.graph = graph
        self.faces = []
        self.rebuild()

    def rebuild(self):
        self._compute_faces()

    def _compute_faces(self):
        rot = self._build_rotation()
        self.faces.clear()
        n = len(self.graph.edges)
        visited: List[List[bool]] = [[False, False] for _ in range(n)]
        for start_ei in range(n):
            for start_dir in (1, -1):
                if visited[start_ei][0 if start_dir == 1 else 1]:
                    continue
                face: List[Tuple[int, int]] = []
                ei, d = start_ei, start_dir
                while True:
                    face.append((ei, d))
                    visited[ei][0 if d == 1 else 1] = True
                    ei, d = self._next_dart(ei, d, rot)
                    if ei == start_ei and d == start_dir:
                        break
                if len(face) >= 2:
                    self.faces.append(face)

    def _build_rotation(self) -> Dict[Node, List[Tuple[int, int]]]:
        rot: Dict[Node, List[Tuple[int, int]]] = defaultdict(list)
        for node in self.graph.nodes:
            incident: List[Tuple[int, int]] = []
            for ei, e in enumerate(self.graph.edges):
                if e[0] == node:
                    incident.append((ei, 1))
                if e[1] == node:
                    incident.append((ei, -1))
            incident.sort(key=lambda x: x[0])
            rot[node] = incident
        return rot

    @staticmethod
    def _next_dart(edge_idx: int, direction: int,
                   rot: Dict[Node, List[Tuple[int, int]]],
                   graph: Graph = None) -> Tuple[int, int]:
        pass

    def _next_dart(self, edge_idx: int, direction: int,
                   rot: Dict[Node, List[Tuple[int, int]]] = None) -> Tuple[int, int]:
        if rot is None:
            rot = self._build_rotation()
        edge = self.graph.edges[edge_idx]
        node = edge[1] if direction == 1 else edge[0]
        node_rot = rot.get(node, [])
        if not node_rot:
            return (edge_idx, -direction)
        for i, (ei, d) in enumerate(node_rot):
            if ei == edge_idx and d == -direction:
                nxt = node_rot[(i + 1) % len(node_rot)]
                return (nxt[0], nxt[1])
        return (edge_idx, -direction)

    def find_face_with_nodes(self, u: Node, v: Node) -> Optional[int]:
        for i, face in enumerate(self.faces):
            ns: Set[Node] = set()
            for ei, d in face:
                e = self.graph.edges[ei]
                ns.add(e[0])
                ns.add(e[1])
            if u in ns and v in ns:
                return i
        return None

    def add_edge(self, u: Node, v: Node, face_idx: int) -> int:
        face = self.faces[face_idx]
        ns: Set[Node] = set()
        for ei, d in face:
            e = self.graph.edges[ei]
            ns.add(e[0])
            ns.add(e[1])
        if u not in ns or v not in ns:
            raise ValueError(f"Nodes {u} or {v} not on face {face_idx}")

        self.graph.add_edge((u, v))
        new_ei = len(self.graph.edges) - 1
        self.rebuild()
        return new_ei

    def remove_edge(self, edge_idx: int) -> None:
        e = self.graph.edges[edge_idx]
        self.graph.remove_edge(e)
        self.rebuild()

    def add_crossing(self, e1_idx: int, e2_idx: int, face_idx: int) -> Node:
        e1 = self.graph.edges[e1_idx]
        e2 = self.graph.edges[e2_idx]
        xid = _cfg.crossing_node_fmt.format(
            count=len(self.graph.crossings), e1=e1_idx, e2=e2_idx,
        )
        while xid in self.graph.nodes:
            xid = f"{xid}_{id(xid)}"

        self.graph.remove_edge(e1)
        self.graph.remove_edge(e2)

        new_edges = [
            (e1[0], xid), (xid, e1[1]),
            (e2[0], xid), (xid, e2[1]),
        ]
        for uu, vv in new_edges:
            self.graph.add_edge((uu, vv))

        self.graph.edge_pairs[xid] = (
            ((e1[0], xid), (xid, e1[1])),
            ((e2[0], xid), (xid, e2[1])),
        )
        self.rebuild()
        return xid

    def remove_crossing(self, crossing_node: Node) -> None:
        if crossing_node not in self.graph.edge_pairs:
            return
        (pair1, pair2) = self.graph.edge_pairs[crossing_node]
        in1, out1 = pair1
        in2, out2 = pair2

        self.remove_edge(self._edge_index(in1))
        self.remove_edge(self._edge_index(out1))
        self.remove_edge(self._edge_index(in2))
        self.remove_edge(self._edge_index(out2))

        self.graph.add_edge((in1[0], out1[1]))
        self.graph.add_edge((in2[0], out2[1]))
        del self.graph.edge_pairs[crossing_node]
        self.rebuild()

    def _edge_index(self, edge: Edge) -> int:
        return self.graph.edges.index(edge)

    def non_adjacent_edge_pairs(self, face_idx: int) -> List[Tuple[int, int]]:
        face = self.faces[face_idx]
        n = len(face)
        pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                ei = face[i][0]
                ej = face[j][0]
                if self._edges_share_vertex(ei, ej):
                    continue
                gap = min(abs(i - j), n - abs(i - j))
                if gap <= 1:
                    continue
                pairs.append((ei, ej))
        return pairs

    def _edges_share_vertex(self, ei: int, ej: int) -> bool:
        e1 = self.graph.edges[ei]
        e2 = self.graph.edges[ej]
        return bool({e1[0], e1[1]} & {e2[0], e2[1]})

    def is_valid(self) -> bool:
        return self.graph.validate_underlying()
