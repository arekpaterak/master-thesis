from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import torch.nn as nn

from core.models.nds.decoders.base import DestroyDecoder
from core.models.nds.encoders.config import PositionalEncodingConfig, SplitEncoderConfig
from core.models.nds.encoders.split import InstanceCache, SplitNDSEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.states.destroy import initial_action_mask


class SplitNDSModel(nn.Module):
    """Destroy model with a split-frequency encoder designed specifically for LNS search.

    Encoding lives in :class:`SplitNDSEncoder`: the static instance is encoded once and
    each LNS iteration conditions on the current solution via fusion.
    """

    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp", "cmst"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: SplitEncoderConfig | None = None,
        positional: PositionalEncodingConfig | None = None,
        decoder: DestroyDecoder,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name
        self.encoder = SplitNDSEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=positional,
        )
        self.decoder = decoder

    def rollout(
        self,
        solution: BaseSolution,
        *,
        cache: InstanceCache | None = None,
        rollouts_per_solution: int,
        deconstruction_size: int,
        greedy: bool = False,
    ) -> Rollout:
        """Rollout from the instance encoding.

        A pure function of ``(solution, cache)``: when no ``cache`` is given the static
        encoding is recomputed from the instance. Reuse across a search is provided by
        :meth:`search_session`, which binds the cache into the callable it yields.
        """
        solution, batch_dims = flatten_instance_batch(solution)
        if cache is None:
            cache = self.encoder.encode_instance(solution.instance)

        return self._rollout_from_cache(
            solution,
            batch_dims,
            cache,
            rollouts_per_solution=rollouts_per_solution,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
        )

    def encode_instance(self, instance: BaseProblemInstance) -> InstanceCache:
        return self.encoder.encode_instance(instance.reshape(-1))

    @contextmanager
    def search_session(self, instance: BaseProblemInstance) -> Iterator[Callable[..., Rollout]]:
        cache = self.encode_instance(instance)

        def step(solution: BaseSolution, **kwargs) -> Rollout:
            return self.rollout(solution, cache=cache, **kwargs)

        yield step

    def _rollout_from_cache(
        self,
        solution: BaseSolution,
        batch_dims: int,
        cache: InstanceCache,
        *,
        rollouts_per_solution: int,
        deconstruction_size: int,
        greedy: bool,
    ) -> Rollout:
        instance = solution.instance

        B = instance.batch_size[0]
        R = rollouts_per_solution
        device = get_model_device(self)

        seed = sample_seed(
            batch_size=B, rollout_size=R, seed_dim=self.decoder.seed_dim, device=device
        )

        h_nodes = self.encoder(cache, solution)

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
