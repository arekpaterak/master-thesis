from __future__ import annotations

from typing import Literal

import torch
from torch import nn

QueryStrategy = Literal["mean", "max", "meanmax"]


class DecoderQuery(nn.Module):
    """Pools node embeddings into a single cross-attention query.

    The query summarizes the graph before it attends over the nodes.
    Three strategies:
    - ``mean``: the mean over nodes.
    - ``max``: the per-feature max over nodes.
    - ``meanmax``: concatenate the mean and the per-feature max, then project back to
      ``hidden_dim`` so the pooled query still matches the attention embed dimension.

    Args:
        hidden_dim: Node embedding dimension (D). The query is always (B, 1, D).
        strategy: Pooling strategy.
    """

    def __init__(self, *, hidden_dim: int, strategy: QueryStrategy = "mean"):
        super().__init__()
        self.strategy = strategy
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim) if strategy == "meanmax" else None

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        # h_nodes: (B, N, D) -> query: (B, 1, D)
        if self.strategy == "mean":
            return h_nodes.mean(dim=-2, keepdim=True)  # (B, 1, D)
        if self.strategy == "max":
            return h_nodes.max(dim=-2, keepdim=True).values  # (B, 1, D)
        mean = h_nodes.mean(dim=-2, keepdim=True)  # (B, 1, D)
        maximum = h_nodes.max(dim=-2, keepdim=True).values  # (B, 1, D)
        return self.proj(torch.cat([mean, maximum], dim=-1))  # (B, 1, D)
