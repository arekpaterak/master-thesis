import torch
import torch.nn as nn

from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding
from core.models.nds.encoders.config import EncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.layers import (
    AttentionLayer,
    NeighboursEncodingLayer,
    RouteEncodingLayer,
)


class NDSEncoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: EncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        with_neighbours_encoding: bool = True,
        with_route_encoding: bool = True,
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
            self.initial_embedding = initial_embedding
        elif problem_name == "cvrp":
            self.initial_embedding = CVRPInitialEmbedding(
                hidden_dim=hidden_dim, positional=positional
            )
        elif problem_name == "tsp":
            self.initial_embedding = TSPInitialEmbedding(
                hidden_dim=hidden_dim, positional=positional
            )
        else:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        encoder_config = encoder_config or EncoderConfig()
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
        self.neighbours_encoding = (
            NeighboursEncodingLayer(hidden_dim=hidden_dim) if with_neighbours_encoding else None
        )
        self.route_encoding = (
            RouteEncodingLayer(hidden_dim=hidden_dim, aggregation=encoder_config.route_aggregation)
            if with_route_encoding
            else None
        )
        self.post_attention = nn.Sequential(
            *(attention_layer() for _ in range(encoder_config.num_post_attention_layers))
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
        return h
