from dataclasses import dataclass

import torch

from core.models.entropy import entropy
from core.models.nds.seed_sampler import sample_seed
from core.models.utils import get_model_device
from core.problems.cvrp import CVRPSolution
from core.problems.cvrp.features import CVRPFeatureExtractor
from core.problems.solution import BaseSolution


@dataclass
class NDSRollout:
    actions: torch.Tensor
    """(B, R, M)"""
    log_p_of_actions: torch.Tensor
    """(B, R, M)"""
    entropy: float | None
    entropy_norm: float | None

    def __post_init__(self):
        assert self.actions.size() == self.log_p_of_actions.size()

    @property
    def overall_log_p(self) -> torch.Tensor:
        """(B, R)"""
        return self.log_p_of_actions.sum(dim=-1)


def _rollout_solution(
    model,
    solution: BaseSolution,
    *,
    rollouts_per_solution: int,
    deconstruction_size: int,
    greedy: bool = False,
) -> NDSRollout:
    if not isinstance(solution, CVRPSolution):
        raise NotImplementedError("Only works with CVRPSolution")

    instance = solution.instance

    B = instance.batch_size[0]
    R = rollouts_per_solution

    seed = sample_seed(
        batch_size=B,
        rollout_size=R,
        seed_dim=model.z_dim,
        device=get_model_device(model),
    )

    extractor = CVRPFeatureExtractor()
    features = extractor(solution)
    # Move features to correct device
    device = get_model_device(model)
    features = {k: v.to(device) for k, v in features.items()}

    problem_state, decoder_state, cache = model.pre_forward(
        seed=seed,
        rollout_size=rollouts_per_solution,
        **features,
    )

    selected = []
    log_p_of_selected = []
    entropies = []
    entropies_norm = []

    for _ in range(deconstruction_size):
        step_log_p, gru_hidden_state = model.forward(
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

    mean_entropy = torch.stack(entropies, dim=-1).mean().item()
    mean_entropy_norm = torch.stack(entropies_norm, dim=-1).mean().item()

    return NDSRollout(
        actions=selected,
        log_p_of_actions=torch.stack(log_p_of_selected, dim=-1),
        entropy=mean_entropy,
        entropy_norm=mean_entropy_norm,
    )
