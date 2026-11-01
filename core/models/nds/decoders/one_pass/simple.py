from __future__ import annotations

import torch

from core.envs.destroy import DestroyEnv, DestroyState
from core.models.entropy import entropy
from core.models.layers.attention_scaling import AttentionScalingConfig
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.cross_attention import CrossAttention, CrossAttentionConfig
from core.models.layers.decoder_query import DecoderQuery, DecoderQueryConfig
from core.models.layers.poly_net import PolyNet
from core.models.rollout import Rollout

from .base import OnePassDestroyDecoder
from .utils import log_prob_of_sequence_without_replacement


class SimpleOnePassDecoder(OnePassDestroyDecoder):
    def __init__(
        self,
        *,
        hidden_dim: int,
        seed_dim: int,
        poly_dim: int,
        num_heads: int,
        with_poly: bool = True,
        query: DecoderQueryConfig | None = None,
        cross_attention: CrossAttentionConfig | None = None,
        num_context_nodes: int = 0,
        C: float = 10.0,
        attention_scaling: AttentionScalingConfig | None = None,
    ):
        super().__init__()
        self._seed_dim = seed_dim
        self.with_poly = with_poly
        self.num_context_nodes = num_context_nodes
        self.query = DecoderQuery.from_config(
            query or DecoderQueryConfig(),
            hidden_dim=hidden_dim,
            num_context_nodes=num_context_nodes,
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
        self.final_scorer = CompatibilityScorer(
            hidden_dim=hidden_dim, C=C, use_projections=False, attention_scaling=attention_scaling
        )

    @property
    def seed_dim(self) -> int:
        return self._seed_dim

    def forward(
        self, *, h_nodes: torch.Tensor, seed: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        B, N, D = h_nodes.shape
        R = seed.shape[1]
        k = self.num_context_nodes
        query = self.query(
            h_nodes[:, k:], context_embedded_nodes=h_nodes[:, :k] if k else None
        )  # (B, 1, D)
        query = self.cross_attn(query, h_nodes, action_mask).squeeze(-2)  # (B, D)
        query = query.unsqueeze(1).expand(B, R, D)  # (B, R, D)

        if self.poly_net is not None:
            query = query + self.poly_net(query, seed)

        # Nodes are shared across rollouts: pass them un-expanded and let the scorer fold
        # the R rollouts into its query axis. The mask broadcasts over R via (B, 1, N).
        logits = self.final_scorer(
            context=query, node_embeddings=h_nodes, mask=action_mask.unsqueeze(1)
        )  # (B, R, N)
        return logits

    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        env: DestroyEnv,
        state: DestroyState,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), state.action_mask: (B, 1, N)
        deconstruction_size = env.horizon
        logits = self.forward(
            h_nodes=h_nodes, seed=seed, action_mask=state.action_mask.squeeze(1)
        )  # (B, R, N)

        log_p_detached = logits.detach()
        if greedy:
            selected = log_p_detached.topk(deconstruction_size, dim=-1).indices  # (B, R, M)
        else:
            gumbel = torch.distributions.Gumbel(
                torch.zeros_like(log_p_detached), torch.ones_like(log_p_detached)
            ).sample()
            selected = (
                (log_p_detached + gumbel).topk(deconstruction_size, dim=-1).indices
            )  # (B, R, M)

        _entropy = entropy(logits)
        log_p_of_selected = log_prob_of_sequence_without_replacement(logits, selected)  # (B, R)

        return Rollout.create(
            actions=selected,
            log_prob=log_p_of_selected,
            entropy=_entropy.entropy.detach(),
            entropy_norm=_entropy.entropy_norm.detach(),
        )
