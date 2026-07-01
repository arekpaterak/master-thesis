from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.models.nds.config import EncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.layers import (
    AttentionLayer,
    NeighboursEncodingLayer,
    RouteEncodingLayer,
)
from core.models.nds.encoders.nds import NDSEncoder
from core.models.utils import get_model_device
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.problems.tsp.features import (
    TSPInstanceFeatureExtractor,
    TSPSolutionFeatureExtractor,
)


class SplitNDSEncoder(nn.Module):
    """Split-frequency encoder for LNS search.

    The static instance is encoded once by an attention-only ``NDSEncoder``
    (the solution-aware neighbours/route layers are disabled). Each LNS iteration
    conditions the cached embeddings on the current solution through the positional
    fusion projection and up to three optional stages: neighbours encoding, route
    encoding, and self-attention layers.

    The two-phase API mirrors that flow: :meth:`encode_instance` runs the static,
    solution-independent path once; :meth:`fuse` re-conditions the cached embeddings
    on a solution every step.
    """

    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: EncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        with_fusion_neighbours_encoding: bool = False,
        with_fusion_route_encoding: bool = False,
        num_fusion_attention_layers: int = 0,
    ) -> None:
        super().__init__()

        if with_fusion_route_encoding and problem_name != "cvrp":
            raise ValueError(
                "with_fusion_route_encoding is only supported for problem_name='cvrp'."
            )

        self.problem_name = problem_name
        self._positional = positional or PositionalEncodingConfig()

        match problem_name:
            case "cvrp":
                self.instance_feature_extractor = CVRPInstanceFeatureExtractor(with_positional=True)
                self.solution_feature_extractor = CVRPSolutionFeatureExtractor(with_positional=True)
            case "tsp":
                self.instance_feature_extractor = TSPInstanceFeatureExtractor()
                self.solution_feature_extractor = TSPSolutionFeatureExtractor(with_positional=True)
            case _:
                raise ValueError(f"Unsupported problem_name: {problem_name}")

        # The instance encoder must stay solution-independent: structural layers off
        # and no positional encoding injected at the input (it is injected at fusion).
        self.instance_encoder = NDSEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=None,
            with_neighbours_encoding=False,
            with_route_encoding=False,
        )

        # CVRP fuses IPE + XPE, TSP fuses IPE only (no depot anchor).
        pe_components = 2 if problem_name == "cvrp" else 1
        self.fusion = SolutionFusionEncoder(
            hidden_dim=hidden_dim,
            pe_dim=pe_components * self._positional.dim,
            with_neighbours_encoding=with_fusion_neighbours_encoding,
            with_route_encoding=with_fusion_route_encoding,
            num_attention_layers=num_fusion_attention_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
        )

    def encode_instance(self, instance: BaseProblemInstance) -> InstanceCache:
        """Encode the static, solution-independent instance path once."""
        device = get_model_device(self)
        instance_features = self.instance_feature_extractor(instance).to(device)
        instance_node_embeddings = self.instance_encoder(instance_features, None)
        xpe = self._cross_route_encoding(instance_features)
        return InstanceCache(
            instance_node_embeddings=instance_node_embeddings,
            xpe=xpe,
        )

    def fuse(self, cache: InstanceCache, solution: BaseSolution) -> torch.Tensor:
        """Condition the cached instance embeddings on ``solution``.

        Returns:
            (B, N+offset, hidden_dim) node embeddings ready for the decoder.
        """
        device = get_model_device(self)
        solution_features = self.solution_feature_extractor(solution).to(device)
        ipe = self._in_route_positional_encoding(solution_features)
        positional = ipe if cache.xpe is None else torch.cat([ipe, cache.xpe], dim=-1)
        route_index = solution_features.route_index if self.problem_name == "cvrp" else None
        return self.fusion(
            cache.instance_node_embeddings,
            positional,
            neighbours_index=solution_features.neighbours_index,
            route_index=route_index,
        )

    def _cross_route_encoding(self, instance_features) -> torch.Tensor | None:
        if self.problem_name != "cvrp":
            return None
        return self._positional_encoding(instance_features.depot_angle)

    def _in_route_positional_encoding(self, solution_features) -> torch.Tensor:
        if self.problem_name == "cvrp":
            return self._positional_encoding(solution_features.route_phase)
        return self._positional_encoding(solution_features.tour_phase)

    def _positional_encoding(self, phase: torch.Tensor) -> torch.Tensor:
        config = self._positional
        return circular_sinusoidal_encoding(
            phase,
            dim=config.dim,
            direction_aware=config.direction_aware,
            schedule=config.frequency_schedule,
        )


class SolutionFusionEncoder(nn.Module):
    """Condition cached instance embeddings on the current solution.

    Applies up to three stages in sequence:
    1. Project ``[customer_embedding, positional_encoding]`` back to ``hidden_dim``
       (depot passes through unchanged).
    2. Neighbours encoding -- message-passing from prev/next route neighbours.
    3. Route encoding -- route-mean pooling and update.
    4. Self-attention layers over all nodes.

    Stages 2-4 are optional and share the encoder's norm and activation settings.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        pe_dim: int,
        with_neighbours_encoding: bool = False,
        with_route_encoding: bool = False,
        num_attention_layers: int = 0,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        self.neighbours = (
            NeighboursEncodingLayer(hidden_dim=hidden_dim) if with_neighbours_encoding else None
        )
        self.route = RouteEncodingLayer(hidden_dim=hidden_dim) if with_route_encoding else None
        _config = encoder_config or EncoderConfig()
        self.attention_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    norm_type=_config.norm_type,
                    pre_norm=_config.pre_norm,
                    ff_activation=_config.ff_activation,
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        positional: torch.Tensor,
        *,
        neighbours_index: torch.Tensor | None = None,
        route_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_nodes: (B, N+offset, hidden_dim) cached instance embeddings.
            positional: (B, N, pe_dim) per-customer solution positional encoding.
            neighbours_index: (B, N, 2) required when neighbours encoding is enabled.
            route_index: (B, N) required when route encoding is enabled.

        Returns:
            (B, N+offset, hidden_dim)
        """
        offset = h_nodes.shape[-2] - positional.shape[-2]
        customers = h_nodes[:, offset:]
        fused_customers = self.projection(torch.cat([customers, positional], dim=-1))
        h = torch.cat([h_nodes[:, :offset], fused_customers], dim=1)
        if self.neighbours is not None:
            h = self.neighbours(h, neighbours_index=neighbours_index)
        if self.route is not None:
            h = self.route(h, route_index=route_index)
        for layer in self.attention_layers:
            h = layer(h)
        return h


class InstanceCache(NamedTuple):
    """Static per-instance encoder products reused across LNS iterations.

    Holds the instance embeddings and the cross-route positional encoding (XPE),
    all independent of the current solution.
    """

    instance_node_embeddings: torch.Tensor
    xpe: torch.Tensor | None
