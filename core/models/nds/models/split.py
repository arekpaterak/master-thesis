from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.models.nds.config import EncoderConfig, PositionalEncodingConfig
from core.models.nds.decoders.base import DestroyDecoder
from core.models.nds.encoder import NDSEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
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
from core.states.destroy import initial_action_mask


class SolutionPositionalFusion(nn.Module):
    """Concatenate per-customer solution positional encodings onto cached instance
    embeddings and project back to ``hidden_dim``.

    The depot row (offset) passes through unchanged, mirroring ``CVRPInitialEmbedding``
    which keeps the depot positional-encoding-free.
    """

    def __init__(self, *, hidden_dim: int, pe_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim + pe_dim, hidden_dim)

    def forward(self, h_nodes: torch.Tensor, positional: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_nodes: (B, N+offset, hidden_dim) cached instance embeddings.
            positional: (B, N, pe_dim) per-customer solution positional encoding.

        Returns:
            (B, N+offset, hidden_dim)
        """
        offset = h_nodes.shape[-2] - positional.shape[-2]
        customer = h_nodes[:, offset:]
        fused = self.projection(torch.cat([customer, positional], dim=-1))
        return torch.cat([h_nodes[:, :offset], fused], dim=1)


class InstanceCache(NamedTuple):
    """Static per-instance encoder products reused across LNS iterations.

    Holds the instance embeddings and the cross-route positional encoding (XPE, ``None``
    for TSP) -- all independent of the current solution.
    """

    instance_node_embeddings: torch.Tensor
    xpe: torch.Tensor | None


class SplitNDSModel(nn.Module):
    """Destroy model with a split-frequency encoder designed specifically for LNS search.

    The static instance is encoded once by an attention-only ``NDSEncoder``
    (the solution-aware neighbours/route layers are disabled).
    The current solution enters each iteration only through positional encodings
    concatenated onto the cached instance node embeddings just before the decoder, fused back to ``hidden_dim``.
    At inference the instance encoding is cached across iterations;
    during training it is recomputed each step so the per-step ``backward`` stays valid.
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
        decoder: DestroyDecoder,
    ) -> None:
        super().__init__()

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
        self.encoder = NDSEncoder(
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
        self.fusion = SolutionPositionalFusion(
            hidden_dim=hidden_dim, pe_dim=pe_components * self._positional.dim
        )
        self.decoder = decoder

        self._cached_instance: BaseProblemInstance | None = None
        self._instance_cache: InstanceCache | None = None

    def rollout(
        self,
        solution: BaseSolution,
        *,
        rollouts_per_solution: int,
        deconstruction_size: int,
        greedy: bool = False,
    ) -> Rollout:
        # Identity anchor for the instance cache, taken before flattening reshapes the
        # solution (and its instance) into a fresh object on every call.
        instance_anchor = solution.instance

        solution, batch_dims = flatten_instance_batch(solution)
        instance = solution.instance

        B = instance.batch_size[0]
        R = rollouts_per_solution
        device = get_model_device(self)

        seed = sample_seed(
            batch_size=B, rollout_size=R, seed_dim=self.decoder.seed_dim, device=device
        )

        cache = self._instance_cache_for(instance_anchor, instance)

        solution_features = self.solution_feature_extractor(solution).to(device)
        ipe = self._in_route_positional_encoding(solution_features)
        positional = ipe if cache.xpe is None else torch.cat([ipe, cache.xpe], dim=-1)
        h_nodes = self.fusion(cache.instance_node_embeddings, positional)

        N = h_nodes.shape[1]
        action_mask = initial_action_mask(
            self.problem_name, batch_size=B, num_nodes=N, device=device
        )  # (B, N)

        rollout = self.decoder.rollout(
            h_nodes=h_nodes,
            seed=seed,
            action_mask=action_mask,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
        )
        return unflatten_rollout(rollout, batch_dims)

    def _instance_cache_for(
        self, anchor: BaseProblemInstance, instance: BaseProblemInstance
    ) -> InstanceCache:
        """Return the static instance encoding, recomputing only when needed.

        Training always recomputes: the per-step ``loss.backward`` frees the graph, so
        a tensor reused across steps would break the next backward. At inference (eval
        mode) the encoding is cached and reused while the same instance object drives
        the search.
          Holding a reference to ``anchor`` keeps the comparison sound -- an
        alive instance cannot have its identity reused.
        """
        if self.training:
            return self._encode_instance(instance)

        if anchor is not self._cached_instance:
            self._instance_cache = self._encode_instance(instance)
            self._cached_instance = anchor
        assert self._instance_cache is not None
        return self._instance_cache

    def _encode_instance(self, instance: BaseProblemInstance) -> InstanceCache:
        device = get_model_device(self)
        instance_features = self.instance_feature_extractor(instance).to(device)
        instance_node_embeddings = self.encoder(instance_features, None)
        xpe = self._cross_route_encoding(instance_features)
        return InstanceCache(
            instance_node_embeddings=instance_node_embeddings,
            xpe=xpe,
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
