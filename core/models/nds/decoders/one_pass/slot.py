from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.layers.attention_scaling import AttentionScalingConfig
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.cross_attention import CrossAttention, CrossAttentionConfig
from core.models.layers.decoder_query import DecoderQuery, DecoderQueryConfig
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
from core.models.layers.poly_net import PolyNet
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder


class SlotDecoder(OnePassDestroyDecoder):
    """Destroy decoder using M learned slot queries conditioned on the seed.

    Each slot has a dedicated learned query vector (nn.Parameter) that specialises
    during training. The seed is incorporated via poly_net as a residual after MHA,
    adding stochastic conditioning per rollout - the same pattern as SimpleOnePassDecoder.

    When ``query`` has an active component (a pooling or the context nodes; see
    :class:`DecoderQueryConfig`), a node summary is added to every slot query before
    cross-attention, grounding the otherwise instance-blind learned queries in the
    current graph.

    The M slots score all nodes in one pass, then ``rollout`` samples them sequentially:
    each slot draws from its own softmax over the still-available nodes, and the chosen
    node is masked out for the later slots. This yields M distinct nodes with an
    autoregressive joint log-probability.
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
        cross_attention: CrossAttentionConfig | None = None,
        query: DecoderQueryConfig | None = None,
        with_poly: bool = True,
        num_context_nodes: int = 0,
        C: float = 10.0,
        attention_scaling: AttentionScalingConfig | None = None,
    ):
        super().__init__()
        if slot_self_attention not in {"none", "pre", "post"}:
            raise ValueError(f"Unknown slot_self_attention: {slot_self_attention!r}")
        self._seed_dim = seed_dim
        self.num_slots = num_slots
        self.slot_self_attention = slot_self_attention
        self.with_poly = with_poly
        self.num_context_nodes = num_context_nodes

        self.slot_queries = nn.Parameter(torch.empty(num_slots, hidden_dim))
        nn.init.xavier_uniform_(self.slot_queries.unsqueeze(0))

        # The slot queries are the learned base, so the query only pools/adds context.
        query = query or DecoderQueryConfig(pooling="none")
        self.node_query = (
            DecoderQuery.from_config(
                query, hidden_dim=hidden_dim, num_context_nodes=num_context_nodes
            )
            if query.is_active_for(num_context_nodes=num_context_nodes)
            else None
        )

        self.slot_self_attn = (
            MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
            if slot_self_attention != "none"
            else None
        )
        self.cross_attn = CrossAttention.from_config(
            cross_attention or CrossAttentionConfig(),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_context_nodes=num_context_nodes,
            attention_scaling=attention_scaling,
        )
        self.poly_net = (
            PolyNet(hidden_dim=hidden_dim, seed_dim=seed_dim, poly_dim=poly_dim)
            if with_poly
            else None
        )
        self.scorer = CompatibilityScorer(
            hidden_dim=hidden_dim, C=C, use_projections=False, attention_scaling=attention_scaling
        )

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
        k = self.num_context_nodes

        # MHA and slot self-attention do not depend on the rollout dimension R,
        # so compute them once at (B, num_slots, hidden_dim) and expand to R only
        # before poly_net, where the per-rollout seed first enters.
        queries = self.slot_queries.expand(B, self.num_slots, hidden_dim)  # (B, M, D)

        if self.node_query is not None:
            # Add the node summary (B, 1, D) to every slot, broadcasting over M.
            queries = queries + self.node_query(
                h_nodes[:, k:], context_embedded_nodes=h_nodes[:, :k] if k else None
            )

        if self.slot_self_attention == "pre":
            assert self.slot_self_attn is not None
            queries = queries + self.slot_self_attn(queries, queries, queries)

        queries = self.cross_attn(queries, h_nodes, action_mask)  # (B, M, hidden_dim)

        if self.slot_self_attention == "post":
            assert self.slot_self_attn is not None
            queries = queries + self.slot_self_attn(queries, queries, queries)

        queries = queries.unsqueeze(1).expand(B, R, self.num_slots, hidden_dim)  # (B, R, M, D)
        if self.poly_net is not None:
            seed_for_slots = seed.unsqueeze(-2).expand(B, R, self.num_slots, self.seed_dim)
            queries = queries + self.poly_net(queries, seed_for_slots)  # (B, R, M, hidden_dim)

        # Nodes are shared across rollouts and slots: pass them un-expanded and let the
        # scorer fold (R, M) into its query axes. The mask broadcasts via (B, 1, 1, N).
        # Return raw scores; rollout normalizes each slot over the still-available nodes.
        node_mask = action_mask[:, None, None]  # (B, 1, 1, N)
        return self.scorer(
            context=queries, node_embeddings=h_nodes, mask=node_mask, log_softmax=False
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

        scores = self.forward(
            h_nodes=h_nodes, seed=seed, action_mask=action_mask
        )  # (B, R, M, N) raw, forbidden nodes at -inf
        B, R, M, N = scores.shape

        # Sample slots sequentially, masking each chosen node out of the later slots so the
        # selected nodes are distinct. The joint log-prob is the autoregressive sum.
        already_selected = torch.zeros(B, R, N, dtype=torch.bool, device=scores.device)
        selected, log_probs, entropies, entropies_norm = [], [], [], []
        for slot in range(M):
            log_p = torch.log_softmax(
                scores[:, :, slot, :].masked_fill(already_selected, float("-inf")), dim=-1
            )  # (B, R, N)
            if greedy:
                idx = log_p.argmax(dim=-1)  # (B, R)
            else:
                idx = torch.distributions.Categorical(logits=log_p).sample()  # (B, R)

            selected.append(idx)
            log_probs.append(log_p.gather(-1, idx.unsqueeze(-1)).squeeze(-1))  # (B, R)
            step_entropy = entropy(log_p)
            entropies.append(step_entropy.entropy.detach())
            entropies_norm.append(step_entropy.entropy_norm.detach())
            already_selected = already_selected.scatter(-1, idx.unsqueeze(-1), True)

        return Rollout.create(
            actions=torch.stack(selected, dim=-1),  # (B, R, M)
            log_prob=torch.stack(log_probs, dim=-1).sum(dim=-1),  # (B, R)
            entropy=torch.stack(entropies, dim=-1).mean(dim=-1),
            entropy_norm=torch.stack(entropies_norm, dim=-1).mean(dim=-1),
        )
