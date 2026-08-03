from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.models.nds.encoders.config import (
    EncoderConfig,
    JointEncoderConfig,
    PositionalEncodingConfig,
    SplitEncoderConfig,
)
from core.models.nds.encoders.joint import JointEncoder
from core.models.nds.encoders.layers import (
    AttentionLayer,
    NeighboursEncodingLayer,
    RouteEncodingLayer,
    TreeNeighboursEncodingLayer,
    build_norm,
)
from core.models.utils import get_model_device
from core.problems.cmstp.features import (
    CMSTPInstanceFeatureExtractor,
    CMSTPSolutionFeatureExtractor,
)
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
    solution-independent path once; :meth:`forward` re-conditions the cached embeddings
    on a solution every step.
    """

    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp", "cmstp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: SplitEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
    ) -> None:
        super().__init__()

        encoder_config = encoder_config or SplitEncoderConfig()
        with_route_encoding = encoder_config.with_route_encoding
        if with_route_encoding is None:
            with_route_encoding = problem_name in ("cvrp", "cmstp")
        if with_route_encoding and problem_name not in ("cvrp", "cmstp"):
            raise ValueError(
                "with_route_encoding is only supported for problem_name in {'cvrp', 'cmstp'}."
            )
        if positional is not None and problem_name == "cmstp":
            raise NotImplementedError("CMSTP positional encoding is not implemented yet.")

        self.problem_name = problem_name
        self._positional = positional

        with_positional = self._positional is not None
        match problem_name:
            case "cvrp":
                self.instance_feature_extractor = CVRPInstanceFeatureExtractor(
                    with_positional=with_positional
                )
                self.solution_feature_extractor = CVRPSolutionFeatureExtractor(
                    with_positional=with_positional
                )
            case "tsp":
                self.instance_feature_extractor = TSPInstanceFeatureExtractor()
                self.solution_feature_extractor = TSPSolutionFeatureExtractor(
                    with_positional=with_positional
                )
            case "cmstp":
                self.instance_feature_extractor = CMSTPInstanceFeatureExtractor()
                self.solution_feature_extractor = CMSTPSolutionFeatureExtractor()
            case _:
                raise ValueError(f"Unsupported problem_name: {problem_name}")

        # The instance encoder must stay solution-independent: structural layers off
        # and no positional encoding injected at the input (it is injected at fusion).
        shared_fields = encoder_config.model_dump(
            include=set(EncoderConfig.model_fields) - {"with_final_norm"}
        )
        instance_encoder_config = JointEncoderConfig(
            **shared_fields,
            with_final_norm=False,
            with_neighbours_encoding=False,
            with_route_encoding=False,
            num_pre_attention_layers=encoder_config.num_instance_attention_layers,
            num_post_attention_layers=0,
        )
        self.instance_encoder = JointEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=instance_encoder_config,
            positional=None,
        )

        # CVRP injects IPE + XPE, TSP injects IPE only (no depot anchor). 0 when PE is disabled.
        pe_components = 2 if problem_name == "cvrp" else 1
        pe_dim = pe_components * self._positional.dim if self._positional is not None else 0
        self.fusion = SolutionEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            pe_dim=pe_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config.model_copy(
                update={"with_route_encoding": with_route_encoding}
            ),
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

    def forward(self, cache: InstanceCache, solution: BaseSolution) -> torch.Tensor:
        """Condition the cached instance embeddings on ``solution`` (the per-step path).

        Returns:
            (B, N+offset, hidden_dim) node embeddings ready for the decoder.
        """
        device = get_model_device(self)
        solution_features = self.solution_feature_extractor(solution).to(device)
        ipe = self._in_route_positional_encoding(solution_features)
        if ipe is None:
            positional = None
        else:
            positional = ipe if cache.xpe is None else torch.cat([ipe, cache.xpe], dim=-1)
        if self.problem_name == "cmstp":
            return self.fusion(
                cache.instance_node_embeddings,
                positional,
                num_customers=solution_features.parent_index.shape[-1],
                parent_index=solution_features.parent_index,
                route_index=solution_features.route_index,
            )
        route_index = solution_features.route_index if self.problem_name == "cvrp" else None
        return self.fusion(
            cache.instance_node_embeddings,
            positional,
            num_customers=solution_features.neighbours_index.shape[-2],
            neighbours_index=solution_features.neighbours_index,
            route_index=route_index,
        )

    def _cross_route_encoding(self, instance_features) -> torch.Tensor | None:
        if self.problem_name != "cvrp" or self._positional is None:
            return None
        return self._positional_encoding(instance_features.depot_angle)

    def _in_route_positional_encoding(self, solution_features) -> torch.Tensor | None:
        if self._positional is None:
            return None
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


class SolutionEncoder(nn.Module):
    """Enrich cached instance embeddings with the current solution."""

    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp", "cmstp"] = "cvrp",
        hidden_dim: int,
        pe_dim: int,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: SplitEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        _config = encoder_config or SplitEncoderConfig()
        self.problem_name = problem_name
        self.encoding_order = _config.encoding_order
        self.projection = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        neighbours_layer_cls = (
            TreeNeighboursEncodingLayer if problem_name == "cmstp" else NeighboursEncodingLayer
        )
        self.neighbours = (
            neighbours_layer_cls(
                hidden_dim=hidden_dim,
                norm_type=_config.norm_type,
                pre_norm=_config.pre_norm,
            )
            if _config.with_neighbours_encoding
            else None
        )
        self.route = (
            RouteEncodingLayer(
                hidden_dim=hidden_dim,
                aggregation=_config.route_aggregation,
                norm_type=_config.norm_type,
                pre_norm=_config.pre_norm,
            )
            if _config.with_route_encoding
            else None
        )
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
                for _ in range(_config.num_solution_attention_layers)
            ]
        )
        self.final_norm = (
            build_norm(_config.norm_type, hidden_dim) if _config.with_final_norm else None
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        positional: torch.Tensor | None,
        *,
        num_customers: int,
        neighbours_index: torch.Tensor | None = None,
        parent_index: torch.Tensor | None = None,
        route_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_nodes: (B, N+offset, hidden_dim) cached instance embeddings.
            positional: (B, N, pe_dim) per-customer solution positional encoding, or
                None when positional encoding is disabled.
            num_customers: N, used to split off the depot/non-customer offset.
            neighbours_index: (B, N, 2) required when neighbours encoding is enabled (non-CMSTP).
            parent_index: (B, N) required when neighbours encoding is enabled for CMSTP.
            route_index: (B, N) required when route encoding is enabled.

        Returns:
            (B, N+offset, hidden_dim)
        """
        offset = h_nodes.shape[-2] - num_customers
        customers = h_nodes[:, offset:]
        projection_input = (
            customers if positional is None else torch.cat([customers, positional], dim=-1)
        )
        fused_customers = self.projection(projection_input)
        h = torch.cat([h_nodes[:, :offset], fused_customers], dim=1)

        def apply_neighbours(h: torch.Tensor) -> torch.Tensor:
            if self.neighbours is None:
                return h
            if self.problem_name == "cmstp":
                return self.neighbours(h, parent_index=parent_index)
            return self.neighbours(h, neighbours_index=neighbours_index)

        def apply_route(h: torch.Tensor) -> torch.Tensor:
            if self.route is None:
                return h
            return self.route(h, route_index=route_index)

        if self.encoding_order == "route_neighbours":
            h = apply_route(h)
            h = apply_neighbours(h)
        else:
            h = apply_neighbours(h)
            h = apply_route(h)

        for layer in self.attention_layers:
            h = layer(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        return h


class InstanceCache(NamedTuple):
    """Static per-instance encoder products reused across LNS iterations.

    Holds the instance embeddings and the cross-route positional encoding (XPE),
    all independent of the current solution.
    """

    instance_node_embeddings: torch.Tensor
    xpe: torch.Tensor | None
