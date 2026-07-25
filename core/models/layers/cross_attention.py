from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import nn

from core.models.layers.multihead_attention import MultiheadAttentionWrapper


class CrossAttentionConfig(BaseModel):
    """Configures a decoder's node cross-attention, where the query attends to the nodes."""

    skip: bool = Field(
        default=False,
        title="Skip cross-attention",
        description=(
            "Skip the node cross-attention entirely and feed the query straight to the "
            "scorer, so node scores are read off the un-refined query."
        ),
    )
    attend_to_context: bool = Field(
        default=False,
        title="Attend to context nodes",
        description=(
            "Let the query attend to the non-selectable context nodes (e.g. the CVRP "
            "depot), which the action mask otherwise excludes. The scorer still cannot "
            "select them. No effect when skipped or without context nodes (e.g. TSP)."
        ),
    )
    residual: bool = Field(
        default=False,
        title="Cross-attention residual",
        description=(
            "Add the query back as a residual around the cross-attention, carrying its "
            "identity through the read instead of overwriting it with the attended node "
            "content."
        ),
    )


class CrossAttention(nn.Module):
    """A decoder's node cross-attention: the query attends to the node embeddings.

    Args:
        hidden_dim: Node embedding dimension (D).
        num_heads: Attention heads.
        skip: Return the query unchanged instead of attending.
        attend_to_context: Also attend to the leading ``num_context_nodes`` context nodes,
            which ``action_mask`` otherwise excludes.
        residual: Add the query back around the attention.
        num_context_nodes: Number of leading non-selectable context nodes.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        skip: bool = False,
        attend_to_context: bool = False,
        residual: bool = False,
        num_context_nodes: int = 0,
    ):
        super().__init__()
        self.attend_to_context = attend_to_context
        self.residual = residual
        self.num_context_nodes = num_context_nodes
        self.mha = (
            None if skip else MultiheadAttentionWrapper(embed_dim=hidden_dim, num_heads=num_heads)
        )

    @classmethod
    def from_config(
        cls,
        config: CrossAttentionConfig,
        *,
        hidden_dim: int,
        num_heads: int,
        num_context_nodes: int,
    ) -> CrossAttention:
        return cls(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            skip=config.skip,
            attend_to_context=config.attend_to_context,
            residual=config.residual,
            num_context_nodes=num_context_nodes,
        )

    def forward(
        self, query: torch.Tensor, node_embeddings: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        # query: (B, *q, D), node_embeddings: (B, N, D), action_mask: (B, N) True = forbidden
        if self.mha is None:
            return query
        k = self.num_context_nodes
        attend_mask = ~action_mask  # (B, N), True = attend
        if self.attend_to_context and k:
            attend_mask[:, :k] = True  # also let the query attend to the context nodes
        attended = self.mha(query, node_embeddings, node_embeddings, attention_mask=attend_mask)
        return query + attended if self.residual else attended
