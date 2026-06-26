from __future__ import annotations

from typing import Literal

import torch.nn as nn

from core.models.nds.config import EncoderConfig, PositionalEncodingConfig
from core.models.nds.decoders.one_pass.base import OnePassDestroyDecoder
from core.models.nds.decoders.one_pass.proximity import (
    ProximityCategoricalDecoder,
    ProximityDecoder,
)
from core.models.nds.encoder import NDSEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems.cvrp.features import (
    CVRPInstanceFeatureExtractor,
    CVRPSolutionFeatureExtractor,
)
from core.problems.solution import BaseSolution
from core.problems.tsp.features import (
    TSPInstanceFeatureExtractor,
    TSPSolutionFeatureExtractor,
)
from core.states.destroy import CVRPDestroyState, TSPDestroyState


class OnePassNDSModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: EncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        with_neighbours_encoding: bool = True,
        with_route_encoding: bool | None = None,
        decoder: OnePassDestroyDecoder,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name

        if with_route_encoding is None:
            with_route_encoding = problem_name == "cvrp"

        with_positional = positional is not None

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
            case _:
                raise ValueError(f"Unsupported problem_name: {self.problem_name}")

        self.encoder = NDSEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=positional,
            with_neighbours_encoding=with_neighbours_encoding,
            with_route_encoding=with_route_encoding,
        )
        self.decoder = decoder

    def rollout(
        self,
        solution: BaseSolution,
        *,
        rollouts_per_solution: int,
        deconstruction_size: int,
        greedy: bool = False,
    ) -> Rollout:
        solution, batch_dims = flatten_instance_batch(solution)
        instance = solution.instance

        B = instance.batch_size[0]
        R = rollouts_per_solution
        device = get_model_device(self)

        seed = sample_seed(
            batch_size=B, rollout_size=R, seed_dim=self.decoder.seed_dim, device=device
        )

        instance_features = self.instance_feature_extractor(instance).to(device)
        solution_features = self.solution_feature_extractor(solution).to(device)

        encoded_nodes = self.encoder(instance_features, solution_features)  # (B, N, D)
        N = encoded_nodes.shape[1]

        match self.problem_name:
            case "cvrp":
                destroy_state_cls = CVRPDestroyState
            case "tsp":
                destroy_state_cls = TSPDestroyState

        action_mask = destroy_state_cls.build_initial(
            batch_size=B, rollout_size=1, num_nodes=N, device=device
        ).action_mask.squeeze(1)  # (B, N)

        extra_kwargs = {}
        if isinstance(self.decoder, (ProximityDecoder, ProximityCategoricalDecoder)):
            extra_kwargs["node_coordinates"] = instance_features.locations  # (B, N, 2)

        rollout = self.decoder.rollout(
            h_nodes=encoded_nodes,
            seed=seed,
            action_mask=action_mask,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
            **extra_kwargs,
        )
        return unflatten_rollout(rollout, batch_dims)
