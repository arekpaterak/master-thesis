from typing import NamedTuple

import torch
import torch.nn as nn

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

    def forward(self, instance_features) -> EncoderOutput:
        """
        Args:
            instance_features: Instance feature container consumed by the initial embedding.

        Returns:
            h_nodes: (B, N, hidden_dim)
            h_graph: (B, hidden_dim)
        """
        h = self.initial_embedding(instance_features)

        for layer in self.layers:
            h = layer(h)

        # Graph-level embedding: mean of node embeddings per graph
        h_graph = h.mean(dim=-2)

        return EncoderOutput(h, h_graph)


class AttentionLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int = 8, ff_dim: int = 512) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        # Full self-attention over the nodes of each instance. Projections without biases,
        # as in the reference implementation of Kool et al. (2019).
        self.MHA = nn.MultiheadAttention(hidden_dim, num_heads, bias=False, batch_first=True)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.FF = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)

        Returns:
            (B, N, hidden_dim)
        """
        # MHA sublayer (Eq. 2). need_weights=False routes through the fused
        # scaled_dot_product_attention kernel (the attention weights are unused).
        mha_out, _ = self.MHA(x, x, x, need_weights=False)
        x = self._batch_norm(self.norm1, x + mha_out)

        # FF sublayer (Eq. 3)
        x = self._batch_norm(self.norm2, x + self.FF(x))

        return x

    @staticmethod
    def _batch_norm(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        """Batch normalisation over all nodes of all instances, as in the paper."""
        return norm(x.reshape(-1, x.size(-1))).view_as(x)
