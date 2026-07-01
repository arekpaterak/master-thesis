from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder


class SlotDecoder(OnePassDestroyDecoder):
    """Destroy decoder using M independent learned slot queries conditioned on the seed.

    Each slot has a dedicated learned query vector (nn.Parameter) that specialises
    during training. The seed is incorporated via poly_net as a residual after MHA,
    adding stochastic conditioning per rollout - the same pattern as SimpleOnePassDecoder.

    Slots attend to all nodes independently and each samples one node (Categorical),
    so selections are with replacement and duplicates can occur. The joint
    log-probability is the sum of M independent per-slot log-probabilities.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        seed_dim: int,
        poly_dim: int,
        num_heads: int,
        num_slots: int,
        slot_self_attention: Literal["none", "pre", "post"] = "post",
        with_poly: bool = True,
        C: float = 10.0,
    ):
        super().__init__()
        if slot_self_attention not in {"none", "pre", "post"}:
            raise ValueError(f"Unknown slot_self_attention: {slot_self_attention!r}")
        self._seed_dim = seed_dim
        self.num_slots = num_slots
        self.slot_self_attention = slot_self_attention
        self.with_poly = with_poly

        self.slot_queries = nn.Parameter(torch.empty(num_slots, hidden_dim))
        nn.init.xavier_uniform_(self.slot_queries.unsqueeze(0))

        self.slot_self_attn = (
            MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
            if slot_self_attention != "none"
            else None
        )
        self.cross_attn = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = (
            nn.Sequential(
                nn.Linear(hidden_dim + seed_dim, poly_dim),
                nn.ReLU(),
                nn.Linear(poly_dim, hidden_dim),
            )
            if with_poly
            else None
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
        # h_nodes: (B, N, hidden_dim), seed: (B, R, seed_dim), action_mask: (B, N)
        B, N, hidden_dim = h_nodes.shape
        R = seed.shape[1]

        # MHA and slot self-attention do not depend on the rollout dimension R,
        # so compute them once at (B, num_slots, hidden_dim) and expand to R only
        # before poly_net, where the per-rollout seed first enters.
        queries = self.slot_queries.expand(B, self.num_slots, hidden_dim)  # (B, M, D)
        if self.slot_self_attention == "pre":
            assert self.slot_self_attn is not None
            queries = queries + self.slot_self_attn(queries, queries, queries)

        mha_out = self.cross_attn(
            queries, h_nodes, h_nodes, attention_mask=~action_mask
        )  # (B, M, hidden_dim)

        if self.slot_self_attention == "post":
            assert self.slot_self_attn is not None
            mha_out = mha_out + self.slot_self_attn(mha_out, mha_out, mha_out)

        mha_out = mha_out.unsqueeze(1).expand(B, R, self.num_slots, hidden_dim)  # (B, R, M, D)
        if self.poly_net is None:
            context = mha_out  # (B, R, M, hidden_dim)
        else:
            seed_expanded = seed.unsqueeze(-2).expand(B, R, self.num_slots, self.seed_dim)
            context = mha_out + self.poly_net(
                torch.cat([mha_out, seed_expanded], dim=-1)
            )  # (B, R, M, hidden_dim)

        # Nodes are shared across rollouts and slots: pass them un-expanded and let the
        # scorer fold (R, M) into its query axes. The mask broadcasts via (B, 1, 1, N).
        return self.scorer(
            h_context=context, h_nodes=h_nodes, mask=action_mask[:, None, None]
        )  # (B, R, M, N)

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
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        assert deconstruction_size == self.num_slots

        logits_per_slot = self.forward(
            h_nodes=h_nodes, seed=seed, action_mask=action_mask
        )  # (B, R, M, N)

        if greedy:
            selected = logits_per_slot.argmax(dim=-1)  # (B, R, M)
        else:
            # Sample one node per slot (independent Categorical, with possible duplicates)
            selected = torch.distributions.Categorical(
                logits=logits_per_slot.detach()
            ).sample()  # (B, R, M)

        log_p_of_selected = logits_per_slot.gather(-1, selected.unsqueeze(-1)).squeeze(
            -1
        )  # (B, R, M)

        # Mean entropy across slots, each slot has an independent distribution over N nodes
        per_slot_entropy = entropy(logits_per_slot)  # (B, R, M)

        return Rollout.create(
            actions=selected,
            log_prob=log_p_of_selected.sum(dim=-1),
            entropy=per_slot_entropy.entropy.mean(dim=-1).detach(),
            entropy_norm=per_slot_entropy.entropy_norm.mean(dim=-1).detach(),
        )
