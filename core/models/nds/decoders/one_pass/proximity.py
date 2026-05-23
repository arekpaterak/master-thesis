from __future__ import annotations

import math

import torch
import torch.nn as nn

from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder


class ProximityDecoder(OnePassDestroyDecoder):
    """Destroy decoder using a continuous 2D coordinate as the action.

    Predicts a Gaussian distribution over [0,1]^2: the network outputs the mean mu,
    a coordinate is sampled from N(mu, sigma^2 * I), and the k nearest nodes are
    selected deterministically. log_p is the Gaussian log-density at the sampled point. Gradients flow through mu via the log-density formula.

    Requires ``node_coords`` to be passed as a keyword argument to ``rollout()``.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        seed_dim: int,
        poly_dim: int,
        num_heads: int,
        sigma: float = 0.1,
    ):
        super().__init__()
        self._seed_dim = seed_dim
        self.sigma = sigma
        self.MHA = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )
        self.coord_projection = nn.Linear(hidden_dim, 2)

    @property
    def seed_dim(self) -> int:
        return self._seed_dim

    def forward(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h_nodes: (B, R, N, D), seed: (B, R, seed_dim)
        # action_mask: (B, R, N) - True = forbidden
        # Returns mu: (B, R, 2) - predicted coordinate mean
        query = h_nodes.mean(dim=-2, keepdim=True)  # (B, R, 1, D)
        mha_out = self.MHA(query, h_nodes, h_nodes, attention_mask=~action_mask).squeeze(
            -2
        )  # (B, R, D)
        context = mha_out + self.poly_net(torch.cat([mha_out, seed], dim=-1))
        return self.coord_projection(context)  # (B, R, 2)

    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
        deconstruction_size: int,
        greedy: bool,
        node_coordinates: torch.Tensor,
        **kwargs,
    ) -> Rollout:
        # h_nodes: (B, R, N, D), seed: (B, R, seed_dim), node_coords: (B, R, N, 2)
        B, R, N, _ = h_nodes.shape

        mu = self.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)  # (B, R, 2)

        coordinates = mu.detach() if greedy else (mu + self.sigma * torch.randn_like(mu)).detach()

        # Select k nearest nodes to the sampled coordinate (action_mask -> inf distance)
        distances = (node_coordinates - coordinates.unsqueeze(-2)).norm(dim=-1)  # (B, R, N)
        distances = distances.masked_fill(action_mask, float("inf"))
        selected = distances.topk(deconstruction_size, dim=-1, largest=False).indices  # (B, R, k)

        # log_p = log N(coordinates | mu, sigma^2 * I) - gradient flows through mu
        # For 2D isotropic Gaussian: -||coordinates - mu||^2 / (2*sigma^2) - 2*log(sigma) - log(2*pi)
        diff = coordinates - mu  # (B, R, 2) - coordinates is detached, grad flows through mu
        log_p = (
            -0.5 * (diff / self.sigma).pow(2).sum(dim=-1)
            - 2 * math.log(self.sigma)
            - math.log(2 * math.pi)
        )  # (B, R)

        # Entropy of 2D isotropic Gaussian: log(2*pi*e*sigma^2) - constant for fixed sigma
        gaussian_entropy = math.log(2 * math.pi * math.e * self.sigma**2)
        entropy_val = torch.full((B, R), gaussian_entropy, device=h_nodes.device)

        return Rollout.create(
            actions=selected,
            log_prob=log_p,
            entropy=entropy_val,
            entropy_norm=torch.zeros(B, R, device=h_nodes.device),
        )


class ProximityCategoricalSelector(OnePassDestroyDecoder):
    """Destroy decoder that converts L2 distances from a predicted coordinate to log-probs.

    Alternative to ProximitySelector: instead of treating the coordinate as a continuous
    action, builds a categorical distribution over nodes via log_softmax(-distances^2)
    and applies Gumbel-top-k + Plackett-Luce - the same interface as SimpleOnePassDecoder.

    Requires node_coords to be passed as a keyword argument to ``rollout()``.
    """

    def __init__(self, *, hidden_dim: int, seed_dim: int, poly_dim: int, num_heads: int):
        super().__init__()
        self._seed_dim = seed_dim
        self.MHA = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )
        self.coord_proj = nn.Linear(hidden_dim, 2)

    @property
    def seed_dim(self) -> int:
        return self._seed_dim

    def forward(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
        deconstruction_size: int,
        greedy: bool,
        node_coordinates: torch.Tensor,
        **kwargs,
    ) -> Rollout:
        # pred_coord (B, R, 2) -> squared L2 distances -> log_softmax -> log_p (B, R, N)
        # Gumbel-top-k + Plackett-Luce (same pattern as SimpleOnePassDecoder)
        raise NotImplementedError
