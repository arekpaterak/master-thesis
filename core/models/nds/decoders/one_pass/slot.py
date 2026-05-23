from __future__ import annotations

import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder


class SlotDecoder(OnePassDestroyDecoder):
    """Destroy decoder using k independent learned slot queries conditioned on the seed.

    Each slot has a dedicated learned query vector (nn.Parameter) that specialises
    during training. The seed is incorporated via poly_net as a residual after MHA,
    adding stochastic conditioning per rollout - the same pattern as SimpleOnePassDecoder.

    Slots attend to all nodes independently and each samples one node (Categorical),
    so selections are with replacement and duplicates can occur. The joint
    log-probability is the sum of k independent per-slot log-probabilities.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        seed_dim: int,
        poly_dim: int,
        num_heads: int,
        k: int,
        C: float = 10.0,
    ):
        super().__init__()
        self._seed_dim = seed_dim
        self.k = k

        self.slot_queries = nn.Parameter(torch.empty(k, hidden_dim))
        nn.init.xavier_uniform_(self.slot_queries.unsqueeze(0))

        self.slot_self_attn = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.MHA = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )
        self.scorer = CompatibilityScorer(hidden_dim=hidden_dim, C=C, use_projections=False)

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
        # h_nodes: (B, R, N, hidden_dim), seed: (B, R, seed_dim)
        # action_mask: (B, R, N) - True = forbidden
        B, R, N, hidden_dim = h_nodes.shape

        queries = self.slot_queries.expand(B, R, self.k, hidden_dim)
        # Self-attention among slots so they become aware of each other before attending to nodes
        slot_context = self.slot_self_attn(queries, queries, queries)
        queries = queries + slot_context

        mha_out = self.MHA(
            queries, h_nodes, h_nodes, attention_mask=~action_mask
        )  # (B, R, k, hidden_dim)

        seed_expanded = seed.unsqueeze(-2).expand(B, R, self.k, self.seed_dim)
        context = mha_out + self.poly_net(
            torch.cat([mha_out, seed_expanded], dim=-1)
        )  # (B, R, k, hidden_dim)

        # Expand h_nodes and mask over the k slot dimension so scorer
        # sees matching batch dims: (B, R, k, N, D) and (B, R, k, N)
        h_nodes_per_slot = h_nodes.unsqueeze(-3).expand(B, R, self.k, N, hidden_dim)
        mask_per_slot = action_mask.unsqueeze(-2).expand(B, R, self.k, N)
        return self.scorer(
            h_c=context, h_nodes=h_nodes_per_slot, mask=mask_per_slot
        )  # (B, R, k, N)

    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor,
        deconstruction_size: int,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        assert deconstruction_size == self.k

        logits_per_slot = self.forward(
            h_nodes=h_nodes, seed=seed, action_mask=action_mask
        )  # (B, R, k, N)

        if greedy:
            selected = logits_per_slot.argmax(dim=-1)  # (B, R, k)
        else:
            # Sample one node per slot (independent Categorical, with possible duplicates)
            selected = torch.distributions.Categorical(
                logits=logits_per_slot.detach()
            ).sample()  # (B, R, k)

        log_p_of_selected = logits_per_slot.gather(-1, selected.unsqueeze(-1)).squeeze(
            -1
        )  # (B, R, k)

        # Mean entropy across slots, each slot has an independent distribution over N nodes
        per_slot_entropy = entropy(logits_per_slot)  # (B, R, k)

        return Rollout.create(
            actions=selected,
            log_prob=log_p_of_selected.sum(dim=-1),
            entropy=per_slot_entropy.entropy.mean(dim=-1).detach(),
            entropy_norm=per_slot_entropy.entropy_norm.mean(dim=-1).detach(),
        )
