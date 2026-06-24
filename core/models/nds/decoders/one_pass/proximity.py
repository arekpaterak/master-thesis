from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.entropy import entropy
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder
from .utils import log_prob_of_sequence_without_replacement


class ProximityDecoder(OnePassDestroyDecoder):
    """Destroy decoder that acts through a single continuous 2D coordinate.

    The network predicts a mean mu in [0,1]^2. A coordinate is sampled from the
    isotropic Gaussian N(mu, sigma^2 * I), and the k nodes nearest to it are removed.
    The log-probability of an action is the Gaussian log-density at the sampled point,
    so gradients reach mu only through that density.

    Requires ``node_coordinates`` to be passed as a keyword argument to ``rollout()``.
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
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        # Returns mu: (B, R, 2) - predicted coordinate mean
        B, N, D = h_nodes.shape
        R = seed.shape[1]
        query = h_nodes.mean(dim=-2, keepdim=True)  # (B, 1, D)
        mha_out = self.MHA(query, h_nodes, h_nodes, attention_mask=~action_mask).squeeze(
            -2
        )  # (B, D)
        mha_out = mha_out.unsqueeze(1).expand(B, R, D)  # (B, R, D)
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
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        # node_coordinates: (B, N, 2)
        B, N, _ = h_nodes.shape
        R = seed.shape[1]

        mu = self.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)  # (B, R, 2)

        coordinates = mu.detach() if greedy else (mu + self.sigma * torch.randn_like(mu)).detach()

        # Select k nearest nodes to the sampled coordinate (action_mask -> inf distance)
        node_coordinates_expanded = node_coordinates.unsqueeze(1).expand(B, R, N, 2)
        action_mask_expanded = action_mask.unsqueeze(1).expand(B, R, N)
        distances = (node_coordinates_expanded - coordinates.unsqueeze(-2)).norm(
            dim=-1
        )  # (B, R, N)
        distances = distances.masked_fill(action_mask_expanded, float("inf"))
        selected = distances.topk(deconstruction_size, dim=-1, largest=False).indices  # (B, R, k)

        # log N(coordinates | mu, sigma^2 * I) for a 2D isotropic Gaussian. Only mu carries
        # gradient here, since coordinates is detached.
        diff = coordinates - mu  # (B, R, 2)
        log_p = (
            -0.5 * (diff / self.sigma).pow(2).sum(dim=-1)
            - 2 * math.log(self.sigma)
            - math.log(2 * math.pi)
        )  # (B, R)

        # Entropy of a 2D isotropic Gaussian, constant for fixed sigma.
        gaussian_entropy = math.log(2 * math.pi * math.e * self.sigma**2)
        entropy_val = torch.full((B, R), gaussian_entropy, device=h_nodes.device)

        return Rollout.create(
            actions=selected,
            log_prob=log_p,
            entropy=entropy_val,
            entropy_norm=torch.zeros(B, R, device=h_nodes.device),
        )


class ProximityCategoricalDecoder(OnePassDestroyDecoder):
    """Destroy decoder that turns a predicted coordinate into a categorical over nodes.

    Like ProximityDecoder, the network predicts a 2D coordinate mu, but instead of
    treating it as the action it scores every node with an RBF kernel centred at mu:
    logit_i = -||coord_i - mu||^2 / (2 * sigma^2). The resulting categorical is sampled
    without replacement.

    Because the log-probability depends on mu through every node's distance (the softmax
    denominator) and through the selected nodes (the numerator), mu receives dense
    gradients tied directly to which nodes were chosen and rewarded - the credit
    assignment that ProximityDecoder lacks. The bandwidth sigma is the model's destroy
    radius, predicted per rollout VAE-style alongside mu (a learned log_sigma head), so
    it can adapt to each instance. With ``anisotropic=True`` sigma becomes a per-axis
    pair, making the kernel an axis-aligned ellipse that can stretch the removal region
    differently along x and y.

    Requires ``node_coordinates`` to be passed as a keyword argument to ``rollout()``.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        seed_dim: int,
        poly_dim: int,
        num_heads: int,
        sigma_init: float = 0.1,
        anisotropic: bool = False,
    ):
        super().__init__()
        self._seed_dim = seed_dim
        self._sigma_dim = 2 if anisotropic else 1
        self.MHA = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )
        # Head emits mu (2) followed by log_sigma (1 isotropic, or 2 per-axis). Bias-init
        # the log_sigma outputs so the initial radius is roughly sigma_init.
        self.coord_projection = nn.Linear(hidden_dim, 2 + self._sigma_dim)
        with torch.no_grad():
            self.coord_projection.bias[2:] = math.log(sigma_init)

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
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        # Returns (B, R, 2 + sigma_dim): mu followed by log_sigma.
        B, N, D = h_nodes.shape
        R = seed.shape[1]
        query = h_nodes.mean(dim=-2, keepdim=True)  # (B, 1, D)
        mha_out = self.MHA(query, h_nodes, h_nodes, attention_mask=~action_mask).squeeze(
            -2
        )  # (B, D)
        mha_out = mha_out.unsqueeze(1).expand(B, R, D)  # (B, R, D)
        context = mha_out + self.poly_net(torch.cat([mha_out, seed], dim=-1))
        return self.coord_projection(context)  # (B, R, 2 + sigma_dim)

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
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        # node_coordinates: (B, N, 2)
        B, N, _ = h_nodes.shape
        R = seed.shape[1]

        head = self.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)
        mu = head[..., :2]  # (B, R, 2)
        sigma = head[..., 2:].exp()  # (B, R, sigma_dim), predicted RBF bandwidth

        # RBF logits: closer nodes get higher score. mu and sigma stay attached so log_p
        # carries gradient back to the coordinate head. A 1-dim sigma gives a circular
        # kernel; a 2-dim sigma weights each axis separately (axis-aligned ellipse).
        diff = node_coordinates.unsqueeze(1) - mu.unsqueeze(-2)  # (B, R, N, 2)
        logits = -(diff.pow(2) / (2 * sigma.pow(2)).unsqueeze(-2)).sum(dim=-1)  # (B, R, N)
        action_mask_expanded = action_mask.unsqueeze(1).expand(B, R, N)
        logits = logits.masked_fill(action_mask_expanded, float("-inf"))

        # Select k nodes without replacement: Gumbel-top-k on detached logits.
        logits_detached = logits.detach()
        if greedy:
            selected = logits_detached.topk(deconstruction_size, dim=-1).indices  # (B, R, k)
        else:
            gumbel = torch.distributions.Gumbel(
                torch.zeros_like(logits_detached), torch.ones_like(logits_detached)
            ).sample()
            selected = (logits_detached + gumbel).topk(deconstruction_size, dim=-1).indices

        log_p_of_selected = log_prob_of_sequence_without_replacement(logits, selected)  # (B, R)
        _entropy = entropy(F.log_softmax(logits, dim=-1))

        return Rollout.create(
            actions=selected,
            log_prob=log_p_of_selected,
            entropy=_entropy.entropy.detach(),
            entropy_norm=_entropy.entropy_norm.detach(),
        )
