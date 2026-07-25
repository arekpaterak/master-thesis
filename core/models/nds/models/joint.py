from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import torch.nn as nn

from core.models.nds.decoders.config import AutoregressiveDecoderConfig, DecoderConfig
from core.models.nds.decoders.factory import build_decoder
from core.models.nds.encoders.config import JointEncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.joint import JointEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems import depot_offset
from core.problems.cmst.features import (
    CMSTInstanceFeatureExtractor,
    CMSTSolutionFeatureExtractor,
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
from core.states.destroy import initial_action_mask


class JointNDSModel(nn.Module):
    """NDS joint-encoder model: re-encodes the instance and solution every step, then
    runs a destroy decoder selected by ``decoder``.

    Any :class:`DestroyDecoder` (autoregressive or one-pass) plugs in; the decoder is
    built from its config in ``__init__``, so the model stays agnostic to the selection
    strategy. Positional encoding, if set, is injected before the encoder.
    """

    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp", "cmst"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: JointEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        decoder: DecoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name

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
            case "cmst":
                self.instance_feature_extractor = CMSTInstanceFeatureExtractor()
                self.solution_feature_extractor = CMSTSolutionFeatureExtractor()
            case _:
                raise ValueError(f"Unsupported problem_name: {problem_name}")

        self.encoder = JointEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=positional,
        )
        self.decoder = build_decoder(
            decoder or AutoregressiveDecoderConfig(),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_context_nodes=depot_offset(problem_name),
        )

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

        h_nodes = self.encoder(instance_features, solution_features)  # (B, N, D)
        action_mask = initial_action_mask(
            self.problem_name, batch_size=B, num_nodes=h_nodes.shape[1], device=device
        )  # (B, N)

        rollout = self.decoder.rollout(
            h_nodes=h_nodes,
            seed=seed,
            action_mask=action_mask,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
        )
        return unflatten_rollout(rollout, batch_dims)

    @contextmanager
    def search_session(self, instance: BaseProblemInstance) -> Iterator[Callable[..., Rollout]]:
        def step(solution: BaseSolution, **kwargs) -> Rollout:
            return self.rollout(solution, **kwargs)

        yield step
