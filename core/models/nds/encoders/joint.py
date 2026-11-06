from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from core.models.nds.encoders.config import JointEncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.layers import AttentionLayer, RouteEncodingLayer, build_norm

if TYPE_CHECKING:
    from core.models.nds.components import NDSComponents


class JointEncoder(nn.Module):
    def __init__(
        self,
        *,
        components: NDSComponents,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: JointEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
    ) -> None:
        super().__init__()

        self.initial_embedding = components.initial_embedding(
            hidden_dim=hidden_dim, positional=positional
        )

        encoder_config = encoder_config or JointEncoderConfig()
        with_route_encoding = resolve_route_encoding(encoder_config.with_route_encoding, components)

        attention_layer = lambda: AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            norm_type=encoder_config.norm_type,
            pre_norm=encoder_config.pre_norm,
            ff_activation=encoder_config.ff_activation,
            attention_scaling=encoder_config.attention_scaling,
        )
        self.pre_attention = nn.Sequential(
            *(attention_layer() for _ in range(encoder_config.num_pre_attention_layers))
        )
        self.encoding_order = encoder_config.encoding_order
        self.neighbours_encoding = (
            components.neighbours_layer(
                hidden_dim=hidden_dim,
                norm_type=encoder_config.norm_type,
                pre_norm=encoder_config.pre_norm,
            )
            if encoder_config.with_neighbours_encoding
            else None
        )
        self.route_encoding = (
            RouteEncodingLayer(
                hidden_dim=hidden_dim,
                aggregation=encoder_config.route_aggregation,
                norm_type=encoder_config.norm_type,
                pre_norm=encoder_config.pre_norm,
            )
            if with_route_encoding
            else None
        )
        self.post_attention = nn.Sequential(
            *(attention_layer() for _ in range(encoder_config.num_post_attention_layers))
        )
        self.final_norm = (
            build_norm(encoder_config.norm_type, hidden_dim)
            if encoder_config.with_final_norm
            else None
        )

    def forward(self, instance, solution) -> torch.Tensor:
        """
        Args:
            instance: Problem-specific instance feature container.
            solution: Problem-specific solution feature container.

        Returns:
            (B, N, hidden_dim) or (B, N+1, hidden_dim)
        """
        h = self.initial_embedding(instance, solution)
        h = self.pre_attention(h)

        def apply_neighbours(h: torch.Tensor) -> torch.Tensor:
            if self.neighbours_encoding is None:
                return h
            return self.neighbours_encoding.encode(h, solution)

        def apply_route(h: torch.Tensor) -> torch.Tensor:
            if self.route_encoding is None:
                return h
            return self.route_encoding(h, route_index=solution.route_index)

        if self.encoding_order == "route_neighbours":
            h = apply_route(h)
            h = apply_neighbours(h)
        else:
            h = apply_neighbours(h)
            h = apply_route(h)

        h = self.post_attention(h)

        if self.final_norm is not None:
            h = self.final_norm(h)
        return h


def resolve_route_encoding(requested: bool | None, components: NDSComponents) -> bool:
    if requested is None:
        return components.route_encoding
    if requested and not components.route_encoding:
        raise ValueError(f"Route encoding is not supported for {components.problem.name}.")
    return requested
