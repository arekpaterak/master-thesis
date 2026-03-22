from typing import NamedTuple

import torch
import torch.nn as nn

from core.models.nds.decoder import DecoderCache, DecoderState, NDSDecoder
from core.models.nds.encoder import NDSEncoder
from core.models.nds.preprocessing import build_encoder_input_indexes_from_sequence
from core.models.nds.seed_sampler import sample_seed
from core.models.utils import get_model_device
from core.problems.cvrp import CVRPSolution


class NDSModel(nn.Module):
    def __init__(
        self,
        *,
        d_h: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        d_poly: int = 256,
        z_dim: int = 10,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.encoder = NDSEncoder(d_h=d_h, num_heads=num_heads, d_ff=d_ff)
        self.decoder = NDSDecoder(d_h=d_h, num_heads=num_heads, z_dim=z_dim, d_poly=d_poly)

    def pre_forward(
        self,
        *,
        depot_location: torch.Tensor,
        customer_features: torch.Tensor,
        neighbours_index: torch.Tensor,
        tour_index: torch.Tensor,
        z: torch.Tensor,
        rollout_size: int,
    ) -> tuple[DecoderState, DecoderCache]:
        B, N, D_H = depot_location.shape

        encoded_nodes = self.encoder(
            depot_location=depot_location,
            customer_features=customer_features,
            neighbours_index=neighbours_index,
            tour_index=tour_index,
        )
        cache = self.decoder.build_cache(h_nodes=encoded_nodes, z=z)
        initial_state = self.decoder.build_initial_state(
            B=B,
            rollout_size=rollout_size,
            N=N,
        )
        return initial_state, cache

    def forward(
        self,
        *,
        state: DecoderState,
        cache: DecoderCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(state=state, cache=cache)

    def select(
        self,
        *,
        state: DecoderState,
        cache: DecoderCache,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_p, gru_hidden_state = self.forward(state=state, cache=cache)

        if greedy:
            idx = log_p.argmax(dim=-1)
        else:
            distribution = torch.distributions.Categorical(logits=log_p / temperature)
            idx = distribution.sample()

        return idx, gru_hidden_state


class Rollout(NamedTuple):
    selected: torch.Tensor
    log_p_of_selected: torch.Tensor
    entropy: float | None
    entropy_norm: float | None


def rollout_policy(
    model: NDSModel,
    solution: CVRPSolution,
    *,
    rollouts_per_solution: int,
    deconstruction_size: int,
    greedy: bool = False,
) -> Rollout:
    instance = solution.instance

    B = instance.batch_size[0]
    R = rollouts_per_solution

    neighbours_index, tour_index = build_encoder_input_indexes_from_sequence(
        solution.sequence, num_customers=instance.num_customers
    )

    z = sample_seed(
        B,
        R,
        z_dim=model.z_dim,
        device=get_model_device(model),
    )

    state, cache = model.pre_forward(
        depot_location=instance.depot_location,
        customer_features=instance.customer_features,
        neighbours_index=neighbours_index,
        tour_index=tour_index,
        z=z,
        rollout_size=rollouts_per_solution,
    )

    D_H = cache.h_nodes_expanded.size(-1)

    selected = []
    for m in range(deconstruction_size):
        log_p, gru_hidden_state = model.forward(state=state, cache=cache)

        if greedy:
            idx = log_p.argmax(dim=-1)
        else:
            distribution = torch.distributions.Categorical(logits=log_p)
            idx = distribution.sample()
        selected.append(idx)

        # Transition DecoderState
        new_mask = state.selected_mask.clone()
        new_mask.scatter_(2, idx.unsqueeze(-1), True)
        new_h_last_node = cache.h_nodes_expanded.gather(
            dim=2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, R, 1, D_H)
        ).squeeze(2)
        state = DecoderState(
            h_last_node=new_h_last_node, gru_hidden=gru_hidden_state, selected_mask=new_mask
        )

    # selected = torch.stack(selected, dim=-1)
