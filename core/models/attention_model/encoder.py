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
        d_x: int = 2,
        d_h: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
    ) -> None:
        """
        Args:
            d_x: input feature dimensionality (2 for TSP: x, y coordinates)
            d_h: hidden embedding dimensionality (paper: 128)
            num_heads: attention heads (paper: 8)
            num_layers: number of attention layers N (paper: 3)
            d_ff: FF hidden dimensionality (paper: 512)
        """
        super().__init__()

        self.problem_name = problem_name
        if problem_name == "tsp":
            self.initial_embedding = TSPInitialEmbedding(
                hidden_dim=d_h,
                node_features_dim=d_x,
            )
        elif problem_name == "cvrp":
            self.initial_embedding = CVRPInitialEmbedding(hidden_dim=d_h)
        else:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        # N attention layers
        self.layers = nn.ModuleList(
            [AttentionLayer(d_h=d_h, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)]
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

        # Initial projection
        if self.problem_name == "cvrp":
            if "depot_features" not in features or "customer_features" not in features:
                raise ValueError("CVRP encoder requires depot_features and customer_features.")
            cvrp_nodes = self.initial_embedding(**features)
            h = cvrp_nodes.reshape(-1, cvrp_nodes.size(-1))
        else:
            if "node_features" not in features:
                raise ValueError("TSP encoder requires node_features.")
            tsp_nodes = self.initial_embedding(**features)
            h = tsp_nodes.reshape(-1, tsp_nodes.size(-1))

        for layer in self.layers:
            h = layer(h, edge_index)

        # Graph-level embedding: mean of node embeddings per graph
        h_graph = global_mean_pool(h, batch)  # (B, d_h)

        return EncoderOutput(h, h_graph)


class AttentionLayer(nn.Module):
    def __init__(self, *, d_h: int, num_heads: int = 8, d_ff: int = 512) -> None:
        super().__init__()
        assert d_h % num_heads == 0

        # TODO: Use simple nn.MultiheadAttention instead of torch_geometric's TransformerConv
        self.MHA = TransformerConv(
            in_channels=d_h,
            out_channels=d_h // num_heads,
            heads=num_heads,
            concat=True,
            dropout=0.0,
            bias=True,
        )
        self.BN1 = nn.BatchNorm1d(d_h)
        self.FF = FeedForward(d_h=d_h, d_ff=d_ff)
        self.BN2 = nn.BatchNorm1d(d_h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # MHA sublayer (Eq. 2)
        # TODO: Adjust BN if using nn.MultiheadAttention instead of TransformerConv
        x = self.BN1(x + self.MHA(x, edge_index))

        # FF sublayer (Eq. 3)
        x = self.BN2(x + self.FF(x))

        return x


class FeedForward(nn.Module):
    def __init__(self, *, d_h: int, d_ff: int = 512) -> None:
        """
        Args:
            d_h: The dimension denoted as $h$ in the paper.
            d_ff: The dimension of the hidden layer in the feedforward network. Equal to 512 in the paper.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
