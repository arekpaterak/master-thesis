from typing import Literal

import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.nds.config import EncoderConfig
from core.models.nds.decoders.ar import (
    AutoregressiveNDSDecoder,
    DecoderCache,
    DecoderOutput,
    DecoderState,
)
from core.models.nds.encoder import NDSEncoder
from core.models.nds.seed_sampler import sample_seed
from core.models.nds.utils import flatten_instance_batch, unflatten_rollout
from core.models.rollout import Rollout
from core.models.utils import get_model_device
from core.problems.cvrp.features import CVRPSolutionFeatureExtractor
from core.problems.solution import BaseSolution
from core.problems.tsp.features import TSPSolutionFeatureExtractor
from core.states.destroy import CVRPDestroyState, DestroyState, TSPDestroyState


class AutoregressiveNDSModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        poly_dim: int = 256,
        seed_dim: int = 10,
        encoder_config: EncoderConfig | None = None,
        with_neighbours_encoding: bool = True,
        with_route_encoding: bool | None = None,
        use_hidden_decoding_state: bool = True,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name
        self.seed_dim = seed_dim

        if with_route_encoding is None:
            with_route_encoding = problem_name == "cvrp"

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
            encoder_config=encoder_config,
            with_neighbours_encoding=with_neighbours_encoding,
            with_route_encoding=with_route_encoding,
        )
        self.decoder = AutoregressiveNDSDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            seed_dim=seed_dim,
            poly_dim=poly_dim,
            use_hidden_decoding_state=use_hidden_decoding_state,
        )

    def pre_forward(
        self,
        *,
        seed: torch.Tensor,
        rollout_size: int,
        **features: torch.Tensor,
    ) -> tuple[DestroyState, DecoderState, DecoderCache]:
        encoded_nodes = self.encoder(**features)

        B = encoded_nodes.shape[0]
        N = encoded_nodes.shape[1]

        destroy_state_cls: type[DestroyState]
        if self.problem_name == "cvrp":
            destroy_state_cls = CVRPDestroyState
        elif self.problem_name == "tsp":
            destroy_state_cls = TSPDestroyState
        else:
            raise ValueError(f"Unsupported problem_name: {self.problem_name}")

        cache = self.decoder.build_cache(h_nodes=encoded_nodes, seed=seed)
        problem_state = destroy_state_cls.build_initial(
            batch_size=B,
            rollout_size=rollout_size,
            num_nodes=N,
            device=encoded_nodes.device,
        )
        initial_state = self.decoder.build_initial_state(
            batch_size=B,
            rollout_size=rollout_size,
        )
        return problem_state, initial_state, cache

    def forward(
        self,
        *,
        problem_state: DestroyState,
        decoder_state: DecoderState,
        cache: DecoderCache,
    ) -> DecoderOutput:
        return self.decoder.forward(
            problem_state=problem_state, decoder_state=decoder_state, cache=cache
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
            batch_size=B,
            rollout_size=R,
            seed_dim=self.seed_dim,
            device=device,
        )

        features = self.feature_extractor(solution)
        # Move features to correct device
        features = {k: v.to(device) for k, v in features.items()}

        problem_state, decoder_state, cache = self.pre_forward(
            seed=seed,
            rollout_size=rollouts_per_solution,
            **features,
        )

        selected = []
        log_prob_of_selected = []
        entropies = []
        entropies_norm = []

        for _ in range(deconstruction_size):
            logits, gru_hidden_state = self.forward(
                problem_state=problem_state,
                decoder_state=decoder_state,
                cache=cache,
            )

            distribution = torch.distributions.Categorical(logits=logits)
            idx = logits.argmax(dim=-1) if greedy else distribution.sample()
            selected.append(idx)

            step_log_prob_of_selected = distribution.log_prob(idx)
            log_prob_of_selected.append(step_log_prob_of_selected)

            step_entropy = entropy(logits)
            entropies.append(step_entropy.entropy.detach())
            entropies_norm.append(step_entropy.entropy_norm.detach())

            problem_state = problem_state.transition(action=idx.unsqueeze(-1))
            decoder_state = decoder_state.transition(gru_hidden_state)

        selected = torch.stack(selected, dim=-1).detach()

        rollout = Rollout.create(
            actions=selected,
            log_prob=torch.stack(log_prob_of_selected, dim=-1).sum(dim=-1),
            entropy=torch.stack(entropies, dim=-1).mean(dim=-1),
            entropy_norm=torch.stack(entropies_norm, dim=-1).mean(dim=-1),
        )
        return unflatten_rollout(rollout, batch_dims)
