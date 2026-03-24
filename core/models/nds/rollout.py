from dataclasses import dataclass

import torch

from core.models.entropy import entropy
from core.models.nds.decoder import DecoderState
from core.models.nds.model import NDSModel
from core.models.nds.preprocessing import build_encoder_input_indexes_from_sequence
from core.models.nds.seed_sampler import sample_seed
from core.models.utils import get_model_device
from core.problems.cvrp import CVRPSolution


@dataclass
class Rollout:
    """
    Attributes:
        actions: A sequence of actions that specify the indices of the customers to be removed.
        log_p_of_actions: A sequence of log probabilities of the actions.
        entropy: A mean entropy of the distributions at each step of the rollout, averaged over the rollout steps and batch.
        entropy_norm: A mean normalized entropy.
    """

    actions: torch.Tensor
    log_p_of_actions: torch.Tensor
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

    seed = sample_seed(
        batch_size=B,
        rollout_size=R,
        seed_dim=model.z_dim,
        device=get_model_device(model),
    )

    state, cache = model.pre_forward(
        depot_location=instance.depot_location,
        customer_features=instance.customer_features,
        neighbours_index=neighbours_index,
        tour_index=tour_index,
        seed=seed,
        rollout_size=rollouts_per_solution,
    )

    selected = []
    log_p_of_selected = []
    entropies = []
    entropies_norm = []

    for m in range(deconstruction_size):
        step_log_p, gru_hidden_state = model.forward(state=state, cache=cache)

        if greedy:
            idx = step_log_p.argmax(dim=-1)
        else:
            distribution = torch.distributions.Categorical(logits=step_log_p)
            idx = distribution.sample()
        selected.append(idx)

        step_log_p_of_selected = step_log_p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        log_p_of_selected.append(step_log_p_of_selected)

        step_entropy = entropy(step_log_p)
        entropies.append(step_entropy.entropy)
        entropies_norm.append(step_entropy.entropy_norm)

        # Transition DecoderState, TODO: Encapsulate
        new_mask = state.selected_mask.clone()
        new_mask.scatter_(2, idx.unsqueeze(-1), True)
        new_h_last_node = cache.h_nodes_expanded.gather(
            dim=2,
            index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, R, 1, cache.h_nodes_expanded.size(-1)),
        ).squeeze(2)
        state = DecoderState(
            h_last_node=new_h_last_node, gru_hidden=gru_hidden_state, selected_mask=new_mask
        )

    selected = torch.stack(selected, dim=-1)

    mean_entropy = torch.stack(entropies, dim=-1).mean().item()
    mean_entropy_norm = torch.stack(entropies_norm, dim=-1).mean().item()

    return Rollout(
        actions=selected,
        log_p_of_actions=torch.stack(log_p_of_selected, dim=-1),
        entropy=mean_entropy,
        entropy_norm=mean_entropy_norm,
    )
