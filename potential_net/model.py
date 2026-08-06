import os
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from tqdm import tqdm

from nn_surrogate.data import NUM_NODE_TYPES
from rf_surrogate.features import FEATURE_DIM


class TargetConditionalGNN(nn.Module):
    def __init__(
        self,
        conv_type: Literal["GCN", "GAT"] = "GCN",
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_sigmoid: bool = True,
        hand_dim: int = 0,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.output_sigmoid = output_sigmoid
        self.hand_dim = hand_dim

        Conv = GCNConv if conv_type == "GCN" else GATConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(Conv(hidden_dim, hidden_dim))

        combined_dim = hidden_dim + 2 + hand_dim
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.head_flow = nn.Linear(hidden_dim, 1)

        head_modules: List[nn.Module] = [nn.Linear(hidden_dim // 2, 1)]
        if output_sigmoid:
            head_modules.append(nn.Sigmoid())
        self.head_short = nn.Sequential(*head_modules)

        head_modules_m: List[nn.Module] = [nn.Linear(hidden_dim // 2, 1)]
        if output_sigmoid:
            head_modules_m.append(nn.Sigmoid())
        self.head_medium = nn.Sequential(*head_modules_m)

    def forward(
        self,
        data: Data,
        target_p: Optional[torch.Tensor] = None,
        target_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.node_emb(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index).relu()
        graph_emb = global_mean_pool(x, data.batch)

        flow_pred: torch.Tensor = self.head_flow(graph_emb).squeeze(-1)

        if target_p is None:
            target_p = data.p
        if target_q is None:
            target_q = data.q
            
        assert target_p is not None
        assert target_q is not None

        if target_p.dim() == 2:
            target_p = target_p.squeeze(-1)
        if target_q.dim() == 2:
            target_q = target_q.squeeze(-1)

        target_feat = torch.stack([target_p, target_q], dim=1).float()
        if self.hand_dim > 0 and hasattr(data, 'hand') and data.hand is not None:
            hand_feat = data.hand.float()
            hand_feat = hand_feat.view(-1, self.hand_dim)
            combined = torch.cat([graph_emb, target_feat, hand_feat], dim=1)
        else:
            combined = torch.cat([graph_emb, target_feat], dim=1)
        h = self.shared(combined)
        return flow_pred, self.head_short(h).squeeze(-1), self.head_medium(h).squeeze(-1)


class PotentialNet:
    def __init__(
        self,
        conv_type: Literal["GCN", "GAT"] = "GCN",
        hidden_dim: int = 128,
        num_layers: int = 5,
        learning_rate: float = 3e-4,
        epochs: int = 300,
        batch_size: int = 256,
        early_stop_patience: int = 30,
        output_sigmoid: bool = True,
        hand_dim: int = 16,
    ) -> None:
        self._conv_type = conv_type
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_sigmoid = output_sigmoid
        self._hand_dim = hand_dim
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = early_stop_patience

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[TargetConditionalGNN] = None
        self._fitted = False

    def _init_model(self) -> None:
        self._model = TargetConditionalGNN(
            conv_type=self._conv_type, # type: ignore
            hidden_dim=self._hidden_dim,
            num_layers=self._num_layers,
            output_sigmoid=self._output_sigmoid,
            hand_dim=self._hand_dim,
        ).to(self._device)

    def _get_target_pq(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        p = batch.p
        q = batch.q
        if p.dim() == 2:
            p = p.squeeze(-1)
        if q.dim() == 2:
            q = q.squeeze(-1)
        return p.float(), q.float()

    @staticmethod
    def pairwise_ranking_loss(
        scores: torch.Tensor,
        true_labels: torch.Tensor,
        margin: float = 0.05,
    ) -> torch.Tensor:
        diff = scores.unsqueeze(0) - scores.unsqueeze(1)
        label_diff = true_labels.unsqueeze(0) - true_labels.unsqueeze(1)
        mask = (label_diff > 0).float()
        n_pairs = mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=scores.device)
        loss = torch.clamp(margin - diff, min=0) * mask
        return loss.sum() / n_pairs

    def fit(
        self,
        data_list: List[Data],
        y_short: np.ndarray,
        y_medium: np.ndarray,
        margin: float = 0.05,
        head_short_weight: float = 0.4,
        head_medium_weight: float = 0.6,
        sample_weights: Optional[np.ndarray] = None,
        reg_weight: float = 1.0,
        loss_type: str = "mse",
        y_flow: Optional[np.ndarray] = None,
        flow_weight: float = 1.0,
    ) -> None:
        n = len(data_list)
        if n < 8:
            return

        for i, d in enumerate(data_list):
            if not hasattr(d, 'y_short') or d.y_short is None:
                d.y_short = FloatTensor([y_short[i]])
            if not hasattr(d, 'y_medium') or d.y_medium is None:
                d.y_medium = FloatTensor([y_medium[i]])
            if y_flow is not None and (not hasattr(d, 'y_flow') or d.y_flow is None):
                d.y_flow = FloatTensor([y_flow[i]])

        indices = list(range(n))
        split = max(8, int(n * 0.8))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_dataset = [data_list[i] for i in train_indices]
        val_dataset = [data_list[i] for i in val_indices]

        if sample_weights is not None:
            w_train = sample_weights[:split]
            w_train = np.maximum(w_train, 1e-8)
            sampler = WeightedRandomSampler(
                w_train.tolist(), num_samples=len(train_dataset), replacement=True,
            )
            train_loader = PyGDataLoader(
                train_dataset, batch_size=self._batch_size, sampler=sampler,
                pin_memory=True,
                pin_memory_device="cuda" if torch.cuda.is_available() else "",
            )
        else:
            train_loader = PyGDataLoader(
                train_dataset, batch_size=self._batch_size, shuffle=True,
                pin_memory=True,
                pin_memory_device="cuda" if torch.cuda.is_available() else "",
            )
        val_loader = PyGDataLoader(
            val_dataset, batch_size=self._batch_size, shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda" if torch.cuda.is_available() else "",
        ) if len(val_dataset) > 0 else None

        self._init_model()
        assert self._model is not None
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        best_val_loss = float("inf")
        best_state: Optional[dict] = None
        patience_counter = 0

        pbar = tqdm(range(self._epochs), desc="Training", unit="ep", dynamic_ncols=True)
        for epoch in pbar:
            self._model.train()
            total_loss = 0.0
            total_graphs = 0
            t0 = time.perf_counter()

            for batch in train_loader:
                batch = batch.to(self._device)
                p, q = self._get_target_pq(batch)
                optimizer.zero_grad()
                flow_pred, s_short, s_medium = self._model(batch, target_p=p, target_q=q)

                loss_short = self.pairwise_ranking_loss(
                    s_short, batch.y_short.squeeze(-1), margin=margin,
                )
                loss_medium = self.pairwise_ranking_loss(
                    s_medium, batch.y_medium.squeeze(-1), margin=margin,
                )
                loss_rank = head_short_weight * loss_short + head_medium_weight * loss_medium

                loss_flow = 0.0
                if y_flow is not None and hasattr(batch, 'y_flow') and batch.y_flow is not None:
                    loss_flow = F.mse_loss(flow_pred, batch.y_flow.squeeze(-1))

                if loss_type == "bce":
                    loss_reg = F.binary_cross_entropy(s_short, batch.y_short.squeeze(-1)) \
                             + F.binary_cross_entropy(s_medium, batch.y_medium.squeeze(-1))
                else:
                    loss_reg = F.mse_loss(s_short, batch.y_short.squeeze(-1)) \
                             + F.mse_loss(s_medium, batch.y_medium.squeeze(-1))
                loss = loss_rank + reg_weight * loss_reg + flow_weight * loss_flow

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs

            avg_train_loss = total_loss / max(total_graphs, 1)

            if val_loader is not None:
                self._model.eval()
                total_val_loss = 0.0
                val_graphs = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self._device)
                        p, q = self._get_target_pq(batch)
                        flow_pred, s_short, s_medium = self._model(batch, target_p=p, target_q=q)
                        loss_short = self.pairwise_ranking_loss(
                            s_short, batch.y_short.squeeze(-1), margin=margin,
                        )
                        loss_medium = self.pairwise_ranking_loss(
                            s_medium, batch.y_medium.squeeze(-1), margin=margin,
                        )
                        loss_rank = head_short_weight * loss_short + head_medium_weight * loss_medium
                        loss_flow = 0.0
                        if y_flow is not None and hasattr(batch, 'y_flow') and batch.y_flow is not None:
                            loss_flow = F.mse_loss(flow_pred, batch.y_flow.squeeze(-1))
                        if loss_type == "bce":
                            loss_reg = F.binary_cross_entropy(s_short, batch.y_short.squeeze(-1)) \
                                     + F.binary_cross_entropy(s_medium, batch.y_medium.squeeze(-1))
                        else:
                            loss_reg = F.mse_loss(s_short, batch.y_short.squeeze(-1)) \
                                     + F.mse_loss(s_medium, batch.y_medium.squeeze(-1))
                        loss = loss_rank + reg_weight * loss_reg + flow_weight * loss_flow
                        total_val_loss += loss.item() * batch.num_graphs
                        val_graphs += batch.num_graphs

                avg_val_loss = total_val_loss / max(val_graphs, 1)

                if avg_val_loss < best_val_loss - 1e-8:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                dt = time.perf_counter() - t0
                pbar.set_postfix_str(
                    f"tr={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                    f"best={best_val_loss:.4f} t={dt:.1f}s"
                )
                if patience_counter >= self._patience:
                    pbar.close()
                    break
            else:
                dt = time.perf_counter() - t0
                pbar.set_postfix_str(f"tr={avg_train_loss:.4f} t={dt:.1f}s")
                if best_state is None:
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

        pbar.close()

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._fitted = True

    def predict(
        self,
        data_list: List[Data],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted or self._model is None:
            raise RuntimeError("PotentialNet not fitted")
        self._model.eval()
        loader = PyGDataLoader(data_list, batch_size=self._batch_size, shuffle=False)
        preds_flow: List[float] = []
        preds_short: List[float] = []
        preds_medium: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                p, q = self._get_target_pq(batch)
                flow_pred, s_short, s_medium = self._model(batch, target_p=p, target_q=q)
                preds_flow.extend(flow_pred.cpu().tolist())
                preds_short.extend(s_short.cpu().tolist())
                preds_medium.extend(s_medium.cpu().tolist())
        return np.array(preds_flow, dtype=np.float64), \
               np.array(preds_short, dtype=np.float64), \
               np.array(preds_medium, dtype=np.float64)

    def score_single(
        self,
        data: Data,
    ) -> Tuple[float, float, float]:
        flows, shorts, mediums = self.predict([data])
        return float(flows[0]), float(shorts[0]), float(mediums[0])

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
                "hand_dim": self._hand_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._conv_type = checkpoint.get("conv_type", self._conv_type)
        self._hidden_dim = checkpoint.get("hidden_dim", self._hidden_dim)
        self._num_layers = checkpoint.get("num_layers", self._num_layers)
        self._output_sigmoid = checkpoint.get("output_sigmoid", self._output_sigmoid)
        self._hand_dim = checkpoint.get("hand_dim", 0)
        self._init_model()
        assert self._model is not None
        self._model.load_state_dict(checkpoint["state_dict"])
        self._fitted = True
