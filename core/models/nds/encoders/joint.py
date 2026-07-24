import torch
import torch.nn as nn

from core.models.embeddings.cmst import CMSTInitialEmbedding
from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding
from core.models.nds.encoders.config import JointEncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.layers import (
    AttentionLayer,
    NeighboursEncodingLayer,
    RouteEncodingLayer,
    TreeNeighboursEncodingLayer,
    build_norm,
)


class JointEncoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: JointEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        initial_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name
        if initial_embedding is not None:
            if problem_name == "cvrp" and not isinstance(initial_embedding, CVRPInitialEmbedding):
                raise TypeError(
                    f"Expected CVRPInitialEmbedding for problem_name='cvrp', got {type(initial_embedding).__name__}"
                )
            if problem_name == "tsp" and not isinstance(initial_embedding, TSPInitialEmbedding):
                raise TypeError(
                    f"Expected TSPInitialEmbedding for problem_name='tsp', got {type(initial_embedding).__name__}"
                )
            if problem_name == "cmst" and not isinstance(initial_embedding, CMSTInitialEmbedding):
                raise TypeError(
                    f"Expected CMSTInitialEmbedding for problem_name='cmst', got {type(initial_embedding).__name__}"
                )
            self.initial_embedding = initial_embedding
        elif problem_name == "cvrp":
            self.initial_embedding = CVRPInitialEmbedding(
                hidden_dim=hidden_dim, positional=positional
            )
        elif problem_name == "tsp":
            self.initial_embedding = TSPInitialEmbedding(
                hidden_dim=hidden_dim, positional=positional
            )
        elif problem_name == "cmst":
            self.initial_embedding = CMSTInitialEmbedding(
                hidden_dim=hidden_dim, positional=positional
            )
        else:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        encoder_config = encoder_config or JointEncoderConfig()
        with_route_encoding = encoder_config.with_route_encoding
        if with_route_encoding is None:
            with_route_encoding = problem_name in ("cvrp", "cmst")

        attention_layer = lambda: AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            norm_type=encoder_config.norm_type,
            pre_norm=encoder_config.pre_norm,
            ff_activation=encoder_config.ff_activation,
        )
        self.pre_attention = nn.Sequential(
            *(attention_layer() for _ in range(encoder_config.num_pre_attention_layers))
        )
        self.encoding_order = encoder_config.encoding_order
        neighbours_layer_cls = (
            TreeNeighboursEncodingLayer if problem_name == "cmst" else NeighboursEncodingLayer
        )
        self.neighbours_encoding = (
            neighbours_layer_cls(
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
            if self.problem_name == "cmst":
                return self.neighbours_encoding(h, parent_index=solution.parent_index)
            return self.neighbours_encoding(h, neighbours_index=solution.neighbours_index)

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
