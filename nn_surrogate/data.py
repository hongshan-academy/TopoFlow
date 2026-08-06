from typing import Dict, List, Tuple, Optional

import torch
from torch import LongTensor, FloatTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from graph import Edge

NODE_TYPE_MAP: Dict[str, int] = {
    "In": 0,
    "Out": 1,
}
NUM_SPECIAL = len(NODE_TYPE_MAP)

PREFIX_TYPE_MAP: Dict[str, int] = {
    "S2_": 2,
    "S3_": 3,
    "C2_": 4,
    "C3_": 5,
}
NUM_NODE_TYPES = len(NODE_TYPE_MAP) + len(PREFIX_TYPE_MAP) + 1  # +1 for isolated/other


def node_type(node: str) -> int:
    if node in NODE_TYPE_MAP:
        return NODE_TYPE_MAP[node]
    for prefix, t in PREFIX_TYPE_MAP.items():
        if node.startswith(prefix):
            return t
    return NUM_NODE_TYPES - 1


def _edges_to_pyg_data(
    edges_tuple: Tuple[Edge, ...],
    y_value: Optional[float] = None,
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

    if y_value is not None:
        y = FloatTensor([y_value])
    else:
        y = FloatTensor([0.0])

    return Data(x=x, edge_index=ei, y=y)


def build_dataloader(
    samples: List[Tuple[Tuple[Edge, ...], float]],
    batch_size: int = 256,
    shuffle: bool = True,
) -> PyGDataLoader:
    data_list = [_edges_to_pyg_data(edges, y_value) for edges, y_value in samples]
    return PyGDataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
    )


def build_data_list(
    samples: List[Tuple[Tuple[Edge, ...], float]],
) -> List[Data]:
    return [_edges_to_pyg_data(edges, y_value) for edges, y_value in samples]


def build_predict_list(
    edges_list: List[Tuple[Edge, ...]],
) -> List[Data]:
    return [_edges_to_pyg_data(edges) for edges in edges_list]
