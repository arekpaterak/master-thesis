from typing import NamedTuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import global_mean_pool

from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding


class EncoderOutput(NamedTuple):
    h_nodes: torch.Tensor
    h_graph: torch.Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str = "tsp",
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
    ) -> None:
        """
        Args:
            hidden_dim: hidden embedding dimensionality (paper: 128)
            num_heads: attention heads (paper: 8)
            num_layers: number of attention layers N (paper: 3)
            ff_dim: FF hidden dimensionality (paper: 512)
        """
        super().__init__()

        self.problem_name = problem_name
        if problem_name == "tsp":
            self.initial_embedding = TSPInitialEmbedding(
                hidden_dim=hidden_dim,
            )
        elif problem_name == "cvrp":
            self.initial_embedding = CVRPInitialEmbedding(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        self.layers = nn.ModuleList(
            [
                AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        *,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        **features: torch.Tensor,
    ) -> EncoderOutput:
        """
        Args:
            **features: Feature dictionary consumed by the selected initial embedding.
            batch: (total_nodes,)
            edge_index: (2, total_edges)

        Returns:
            h_nodes: (total_nodes, d_model)
            h_graph: (batch_size, d_model)
        """
        h = self.initial_embedding(**features)
        h = h.reshape(-1, h.size(-1))

        for layer in self.layers:
            h = layer(h, edge_index)

        # Graph-level embedding: mean of node embeddings per graph
        h_graph = global_mean_pool(h, batch)  # (B, hidden_dim)

        return EncoderOutput(h, h_graph)


class AttentionLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int = 8, ff_dim: int = 512) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        # TODO: Use simple nn.MultiheadAttention instead of torch_geometric's TransformerConv
        self.MHA = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=0.0,
            bias=True,
        )
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.FF = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.BN2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # MHA sublayer (Eq. 2)
        # TODO: Adjust BN if using nn.MultiheadAttention instead of TransformerConv
        x = self.BN1(x + self.MHA(x, edge_index))

        # FF sublayer (Eq. 3)
        x = self.BN2(x + self.FF(x))

        return x
