from __future__ import annotations

from typing import Literal

import torch.nn as nn

from core.models.at_once import ProximityDecoder
from core.models.at_once.decoder.base import OnePassDestroyDecoder
from core.models.nds.encoder import NDSEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems.cvrp.features import CVRPSolutionFeatureExtractor
from core.problems.solution import BaseSolution
from core.problems.tsp.features import TSPSolutionFeatureExtractor
from core.states.destroy import CVRPDestroyState, TSPDestroyState


class OnePassNDSModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        with_neighbours_encoding: bool = True,
        with_tour_encoding: bool | None = None,
        decoder: OnePassDestroyDecoder,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name

        if with_tour_encoding is None:
            with_tour_encoding = problem_name == "cvrp"

        match problem_name:
            case "cvrp":
                self.feature_extractor = CVRPSolutionFeatureExtractor()
            case "tsp":
                self.feature_extractor = TSPSolutionFeatureExtractor()
            case _:
                raise ValueError(f"Unsupported problem_name: {self.problem_name}")

        self.encoder = NDSEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            with_neighbours_encoding=with_neighbours_encoding,
            with_tour_encoding=with_tour_encoding,
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
        instance = solution.instance
        B = instance.batch_size[0]
        R = rollouts_per_solution
        device = get_model_device(self)

        seed = sample_seed(
            batch_size=B, rollout_size=R, seed_dim=self.decoder.seed_dim, device=device
        )

        features = self.feature_extractor(solution)
        features = {key: value.to(device) for key, value in features.items()}

        encoded_nodes = self.encoder(**features)  # (B, N, D)
        N = encoded_nodes.shape[1]

        match self.problem_name:
            case "cvrp":
                destroy_state_cls = CVRPDestroyState
            case "tsp":
                destroy_state_cls = TSPDestroyState

        action_mask = destroy_state_cls.build_initial(
            batch_size=B, rollout_size=R, num_nodes=N, device=device
        ).action_mask  # (B, R, N)

        h_nodes_expanded = encoded_nodes.unsqueeze(1).expand(B, R, N, -1)  # (B, R, N, D)

        extra_kwargs = {}
        if isinstance(self.decoder, ProximityDecoder):
            node_coords = instance.locations.to(device)  # (B, N, 2)
            node_coords_expanded = node_coords.unsqueeze(1).expand(B, R, N, 2)  # (B, R, N, 2)
            extra_kwargs["node_coords"] = node_coords_expanded

        return self.decoder.rollout(
            h_nodes=h_nodes_expanded,
            seed=seed,
            action_mask=action_mask,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
            **extra_kwargs,
        )
