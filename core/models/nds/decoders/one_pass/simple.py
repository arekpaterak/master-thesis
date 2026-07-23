import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.layers.multihead_attention import MultiheadAttentionWrapper
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
        C: float = 10.0,
    ):
        super().__init__()
        self._seed_dim = seed_dim
        self.with_poly = with_poly
        self.MHA = MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        self.poly_net = (
            nn.Sequential(
                nn.Linear(hidden_dim + seed_dim, poly_dim),
                nn.ReLU(),
                nn.Linear(poly_dim, hidden_dim),
            )
            if with_poly
            else None
        )
        self.final_scorer = CompatibilityScorer(hidden_dim=hidden_dim, C=C, use_projections=False)

    @property
    def seed_dim(self) -> int:
        return self._seed_dim

    def forward(
        self, *, h_nodes: torch.Tensor, seed: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        # h_nodes: (B, N, D), seed: (B, R, seed_dim), action_mask: (B, N)
        B, N, D = h_nodes.shape
        R = seed.shape[1]
        query = h_nodes.mean(dim=-2, keepdim=True)  # (B, 1, D)
        mha_out = self.MHA(query, h_nodes, h_nodes, attention_mask=~action_mask).squeeze(
            -2
        )  # (B, D)
        mha_out = mha_out.unsqueeze(1).expand(B, R, D)  # (B, R, D)

        if self.poly_net is None:
            context = mha_out
        else:
            context = mha_out + self.poly_net(torch.cat([mha_out, seed], dim=-1))

        # Nodes are shared across rollouts: pass them un-expanded and let the scorer fold
        # the R rollouts into its query axis. The mask broadcasts over R via (B, 1, N).
        logits = self.final_scorer(
            context=context, node_embeddings=h_nodes, mask=action_mask.unsqueeze(1)
        )  # (B, R, N)
        return logits

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
        logits = self.forward(h_nodes=h_nodes, seed=seed, action_mask=action_mask)  # (B, R, N)

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
