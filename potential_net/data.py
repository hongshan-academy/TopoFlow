from typing import Dict, List, Optional, Tuple

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from graph import Edge
from nn_surrogate.data import NUM_NODE_TYPES, node_type
from rf_surrogate.features import extract_features, FEATURE_DIM


def _build_graph_data(
    edges_tuple: Tuple[Edge, ...],
    target_p: float,
    target_q: float,
    y_short: Optional[float] = None,
    y_medium: Optional[float] = None,
    y_flow: Optional[float] = None,
) -> Data:
    nodes_set: Dict[str, int] = {}
    edge_index: List[Tuple[int, int]] = []

    for src, dst in edges_tuple:
        for n in (src, dst):
            if n not in nodes_set:
                nodes_set[n] = len(nodes_set)
        edge_index.append((nodes_set[src], nodes_set[dst]))

    sorted_nodes = sorted(nodes_set.items(), key=lambda kv: kv[1])
    x_list = [node_type(name) for name, _ in sorted_nodes]

    x = LongTensor(x_list)
    ei = LongTensor(edge_index).t().contiguous()

    hand_vals = extract_features(edges_tuple)
    hand = FloatTensor(hand_vals)

    data = Data(x=x, edge_index=ei)
    data.p = FloatTensor([target_p])
    data.q = FloatTensor([target_q])
    data.hand = hand

    if y_short is not None:
        data.y_short = FloatTensor([y_short])
    if y_medium is not None:
        data.y_medium = FloatTensor([y_medium])
    if y_flow is not None:
        data.y_flow = FloatTensor([y_flow])

    return data


def build_potential_data(
    edges_tuple: Tuple[Edge, ...],
    target_pq: Tuple[int, int],
    y_short: Optional[float] = None,
    y_medium: Optional[float] = None,
    y_flow: Optional[float] = None,
) -> Data:
    return _build_graph_data(
        edges_tuple,
        target_p=float(target_pq[0]),
        target_q=float(target_pq[1]),
        y_short=y_short,
        y_medium=y_medium,
        y_flow=y_flow,
    )


def build_predict_batch(
    edges_list: List[Tuple[Edge, ...]],
    target_pq: Tuple[int, int],
) -> List[Data]:
    return [
        _build_graph_data(e, target_p=float(target_pq[0]), target_q=float(target_pq[1]))
        for e in edges_list
    ]


class PotentialDataset(Dataset):
    def __init__(
        self,
        samples: List,
    ):
        self._data_list: List[Data] = []
        for s in samples:
            edges = s[0]
            target_pq = s[1]
            y_short = s[2]
            y_medium = s[3]
            y_flow = s[6] if len(s) > 6 else None
            d = _build_graph_data(
                edges,
                target_p=float(target_pq[0]),
                target_q=float(target_pq[1]),
                y_short=y_short,
                y_medium=y_medium,
                y_flow=y_flow,
            )
            self._data_list.append(d)

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]
