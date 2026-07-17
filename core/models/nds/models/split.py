from __future__ import annotations

from typing import Literal

import torch.nn as nn

from core.models.nds.decoders.base import DestroyDecoder
from core.models.nds.encoders.config import EncoderConfig, PositionalEncodingConfig
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
    each LNS iteration conditions on the current solution via positional fusion.
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
        with_fusion_route_encoding: bool | None = None,
        num_fusion_attention_layers: int = 0,
        decoder: DestroyDecoder,
    ) -> None:
        super().__init__()

        if with_fusion_route_encoding is None:
            with_fusion_route_encoding = problem_name == "cvrp"

        self.problem_name = problem_name
        self.encoder = SplitNDSEncoder(
            problem_name=problem_name,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_config=encoder_config,
            positional=positional,
            with_fusion_neighbours_encoding=with_fusion_neighbours_encoding,
            with_fusion_route_encoding=with_fusion_route_encoding,
            num_fusion_attention_layers=num_fusion_attention_layers,
        )
        self.decoder = decoder

        self._cached_instance: BaseProblemInstance | None = None
        self._instance_cache: InstanceCache | None = None

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

        When ``cache`` is given (encode-once training, see ``encode_instance``) it is
        used directly. Otherwise, the static encoding is obtained from
        ``_instance_cache_for``: identity-cached at inference, recomputed in training.
        """
        # Identity anchor for the instance cache, taken before flattening reshapes the
        # solution (and its instance) into a fresh object on every call.
        instance_anchor = solution.instance

        solution, batch_dims = flatten_instance_batch(solution)
        if cache is None:
            cache = self._instance_cache_for(instance_anchor, solution.instance)

        return self._rollout_from_cache(
            solution,
            batch_dims,
            cache,
            rollouts_per_solution=rollouts_per_solution,
            deconstruction_size=deconstruction_size,
            greedy=greedy,
        )

    def encode_instance(self, instance: BaseProblemInstance) -> InstanceCache:
        """Encode the static instance once (with grad) for encode-once training.

        The returned cache is reused across every improvement step of a training step;
        only the solution-dependent positional fusion is recomputed per step. Pair this
        with ``rollout(cache=...)`` and accumulate the boundary gradient so a single
        encoder backward suffices. See ``encode_once_in_training``.
        """
        return self.encoder.encode_instance(instance.reshape(-1))

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
            return self.encoder.encode_instance(instance)

        if anchor is not self._cached_instance:
            self._instance_cache = self.encoder.encode_instance(instance)
            self._cached_instance = anchor
        assert self._instance_cache is not None
        return self._instance_cache
