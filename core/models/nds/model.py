from typing import Literal

import torch
import torch.nn as nn

from ...problems.cvrp.features import CVRPSolutionFeatureExtractor
from ...problems.solution import BaseSolution
from ...problems.tsp.features import TSPSolutionFeatureExtractor
from ...states.destroy import CVRPDestroyState, DestroyState, TSPDestroyState
from ..entropy import entropy
from ..rollout import Rollout
from ..utils import get_model_device
from .decoder import DecoderCache, DecoderOutput, DecoderState, NDSDecoder
from .encoder import NDSEncoder
from .seed_sampler import sample_seed


class NDSModel(nn.Module):
    def __init__(
        self,
        *,
        problem_name: Literal["tsp", "cvrp"],
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        poly_dim: int = 256,
        seed_dim: int = 10,
        with_neighbours_encoding: bool = True,
        with_tour_encoding: bool | None = None,
        use_hidden_decoding_state: bool = True,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name
        self.seed_dim = seed_dim

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
        self.decoder = NDSDecoder(
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
            B=B,
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
        log_p_of_selected = []
        entropies = []
        entropies_norm = []

        for _ in range(deconstruction_size):
            step_log_p, gru_hidden_state = self.forward(
                problem_state=problem_state,
                decoder_state=decoder_state,
                cache=cache,
            )

            if greedy:
                idx = step_log_p.argmax(dim=-1)
            else:
                distribution = torch.distributions.Categorical(logits=step_log_p)
                idx = distribution.sample()
            selected.append(idx)

            step_log_p_of_selected = step_log_p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            log_p_of_selected.append(step_log_p_of_selected)

            step_entropy = entropy(step_log_p)
            entropies.append(step_entropy.entropy.detach())
            entropies_norm.append(step_entropy.entropy_norm.detach())

            problem_state = problem_state.transition(action=idx.unsqueeze(-1))
            decoder_state = decoder_state.transition(gru_hidden_state)

        selected = torch.stack(selected, dim=-1).detach()

        return Rollout.create(
            actions=selected,
            log_p=torch.stack(log_p_of_selected, dim=-1).sum(dim=-1),
            entropy=torch.stack(entropies, dim=-1).mean(dim=-1),
            entropy_norm=torch.stack(entropies_norm, dim=-1).mean(dim=-1),
        )
