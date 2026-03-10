from typing import NamedTuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import global_mean_pool


class EncoderOutput(NamedTuple):
    h_nodes: torch.Tensor
    h_graph: torch.Tensor


class Encoder(nn.Module):
    def __init__(
        self,
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

        # Learned linear projection: h_i^(0) = W^x * x_i + b^x
        self.input_projection = nn.Linear(d_x, d_h)

        # N attention layers
        self.layers = nn.ModuleList(
            [AttentionLayer(d_h, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> EncoderOutput:
        """
        Args:
            x: (total_nodes, d_x)
            batch: (total_nodes,)
            edge_index: (2, total_edges)

        Returns:
            h_nodes: (total_nodes, d_model)
            h_graph: (batch_size, d_model)
        """

        # Initial projection
        h = self.input_projection(x)  # (total_nodes, d_h)

        for layer in self.layers:
            h = layer(h, edge_index)

        # Graph-level embedding: mean of node embeddings per graph
        h_graph = global_mean_pool(h, batch)  # (B, d_h)

        return EncoderOutput(h, h_graph)


class FeedForward(nn.Module):
    def __init__(self, model_hidden_dim: int, d_ff: int = 512) -> None:
        """
        Args:
            model_hidden_dim: The dimension denoted as $h$ in the paper.
            d_ff: The dimension of the hidden layer in the feedforward network. Equal to 512 in the paper.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_hidden_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, model_hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionLayer(nn.Module):
    def __init__(self, d_h: int, num_heads: int = 8, d_ff: int = 512) -> None:
        super().__init__()
        assert d_h % num_heads == 0

        self.MHA = TransformerConv(
            in_channels=d_h,
            out_channels=d_h // num_heads,
            heads=num_heads,
            concat=True,
            dropout=0.0,
            bias=True,
        )
        self.BN1 = nn.BatchNorm1d(d_h)
        self.FF = FeedForward(d_h, d_ff)
        self.BN2 = nn.BatchNorm1d(d_h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # MHA sublayer (Eq. 2)
        x = self.BN1(x + self.MHA(x, edge_index))

        # FF sublayer (Eq. 3)
        x = self.BN2(x + self.FF(x))

        return x
