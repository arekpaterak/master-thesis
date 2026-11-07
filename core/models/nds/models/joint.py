from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch.nn as nn

from core.envs.destroy import DestroyEnv
from core.models.nds.decoders.config import AutoregressiveDecoderConfig, DecoderConfig
from core.models.nds.encoders.config import JointEncoderConfig, PositionalEncodingConfig
from core.models.nds.encoders.joint import JointEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.random import SeededGenerator

if TYPE_CHECKING:
    from core.models.nds.components import NDSComponents


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
        components: NDSComponents,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: JointEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        decoder: DecoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.components = components
        self.instance_feature_extractor, self.solution_feature_extractor = (
            components.feature_extractors(positional is not None)
        )

        self.encoder = JointEncoder(
            components=components,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=positional,
        )
        self.decoder = (decoder or AutoregressiveDecoderConfig()).build(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_context_nodes=components.num_context_nodes,
        )

    def rollout(
        self,
        solution: BaseSolution,
        *,
        rollouts_per_solution: int,
        deconstruction_size: int,
        rng: SeededGenerator,
        greedy: bool = False,
    ) -> Rollout:
        solution, batch_dims = flatten_instance_batch(solution)
        instance = solution.instance

        B = instance.batch_size[0]
        R = rollouts_per_solution
        device = get_model_device(self)

        generator = rng.on(device)
        seed = sample_seed(
            batch_size=B,
            rollout_size=R,
            seed_dim=self.decoder.seed_dim,
            generator=generator,
            device=device,
        )

        instance_features = self.instance_feature_extractor(instance).to(device)
        solution_features = self.solution_feature_extractor(solution).to(device)

        h_nodes = self.encoder(instance_features, solution_features)  # (B, N, D)
        env = DestroyEnv(horizon=deconstruction_size)
        rollout = self.decoder.rollout(
            h_nodes=h_nodes,
            seed=seed,
            env=env,
            state=env.reset(instance, device=device),
            greedy=greedy,
            generator=generator,
        )
        return unflatten_rollout(rollout, batch_dims)

    @contextmanager
    def search_session(self, instance: BaseProblemInstance) -> Iterator[Callable[..., Rollout]]:
        def step(solution: BaseSolution, **kwargs) -> Rollout:
            return self.rollout(solution, **kwargs)

        yield step
