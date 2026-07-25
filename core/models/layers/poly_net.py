from __future__ import annotations

import torch
import torch.nn as nn


class PolyNet(nn.Module):
    def __init__(self, *, hidden_dim: int, seed_dim: int, poly_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        # x: (B, *q, hidden_dim), seed: (B, *q, seed_dim) sharing x's query dims.
        # Returns the conditioning term only; callers add it as a residual to x.
        return self.net(torch.cat([x, seed], dim=-1))
