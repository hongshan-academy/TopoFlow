import os
from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from nn_surrogate.data import NUM_NODE_TYPES, build_dataloader


class GNNRegressor(nn.Module):
    def __init__(
        self,
        conv_type: Literal["GCN", "GAT"] = "GCN",
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.output_sigmoid = output_sigmoid

        Conv = GCNConv if conv_type == "GCN" else GATConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(Conv(hidden_dim, hidden_dim))

        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ]
        if output_sigmoid:
            layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, data: Data) -> torch.Tensor:
        x: torch.Tensor = self.node_emb(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return self.mlp(x).squeeze(-1)


class SurrogateGNN:
    def __init__(
        self,
        conv_type: Literal["GCN", "GAT"] = "GCN",
        hidden_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 3e-4,
        epochs: int = 200,
        batch_size: int = 256,
        early_stop_patience: int = 20,
        output_sigmoid: bool = True,
    ) -> None:
        self._conv_type = conv_type
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_sigmoid = output_sigmoid
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = early_stop_patience

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: GNNRegressor | None = None
        self._fitted = False

    def _init_model(self) -> None:
        self._model = GNNRegressor(
            conv_type=self._conv_type, # type: ignore
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            output_sigmoid=self._output_sigmoid,
        ).to(self._device)

    def fit(
        self,
        samples: List[tuple],
        y_values: np.ndarray,
        cached_data: list | None = None,
    ) -> None:
        if len(samples) < 4:
            return

        if cached_data is not None:
            combined = cached_data
            train_loader = PyGDataLoader(
                combined[:max(4, int(len(combined) * 0.8))],
                batch_size=self._batch_size, shuffle=True,
                pin_memory=True,
                pin_memory_device="cuda" if torch.cuda.is_available() else "",
            )
            val_start = max(4, int(len(combined) * 0.8))
            val_loader = PyGDataLoader(
                combined[val_start:],
                batch_size=self._batch_size, shuffle=False,
                pin_memory=True,
                pin_memory_device="cuda" if torch.cuda.is_available() else "",
            )
            n_val = len(combined) - val_start
        else:
            combined = list(zip(samples, y_values))
            n_train = max(4, int(len(combined) * 0.8))
            train_samples = combined[:n_train]
            val_samples = combined[n_train:]

            train_loader = build_dataloader(train_samples, batch_size=self._batch_size, shuffle=True)
            val_loader = build_dataloader(val_samples, batch_size=self._batch_size, shuffle=False)

        self._init_model()
        assert self._model is not None
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        for epoch in range(self._epochs):
            self._model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                pred = self._model(batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_train_loss / max(len(train_loader.dataset), 1)  # type: ignore[arg-type]

            self._model.eval()
            total_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self._device)
                    pred = self._model(batch)
                    loss = loss_fn(pred, batch.y)
                    total_val_loss += loss.item() * batch.num_graphs
                    val_count += batch.num_graphs

            if val_count == 0:
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                break

            avg_val_loss = total_val_loss / val_count

            if avg_val_loss < best_val_loss - 1e-8:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._fitted = True

    def predict(self, data_list: List[Data]) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("SurrogateGNN not fitted")
        self._model.eval()
        loader = PyGDataLoader(data_list, batch_size=self._batch_size, shuffle=False)
        preds: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                out = self._model(batch)
                preds.extend(out.cpu().tolist())
        return np.array(preds, dtype=np.float64)

    def is_ready(self) -> bool:
        return self._fitted

    def save(self, path: str) -> None:
        if self._model is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "conv_type": self._conv_type,
                "hidden_dim": self._hidden_dim,
                "num_layers": self._num_layers,
                "output_sigmoid": self._output_sigmoid,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._conv_type = checkpoint.get("conv_type", self._conv_type)
        self._hidden_dim = checkpoint.get("hidden_dim", self._hidden_dim)
        self._num_layers = checkpoint.get("num_layers", self._num_layers)
        self._output_sigmoid = checkpoint.get("output_sigmoid", self._output_sigmoid)
        self._init_model()
        assert self._model is not None
        self._model.load_state_dict(checkpoint["state_dict"])
        self._fitted = True
