from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.optim as optim

from core.models.nds.model import NDSModel
from core.models.nds.rollout import NDSRollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class EpochSummary(NamedTuple):
    epoch: int
    loss: float
    reward: float


def train_epoch(
    *,
    epoch: int,
    model: NDSModel,
    optimizer: optim.Optimizer,
    num_iterations: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    generate_instance: Callable[[], BaseProblemInstance],
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution],
    apply_rollout: Callable[[BaseSolution, NDSRollout], BaseSolution],
    repair: Callable[[BaseSolution, NDSRollout], BaseSolution],
) -> EpochSummary:
    model.train()
    optimizer.zero_grad()

    instance = generate_instance()
    solution = generate_start_solution(instance)

    for _ in range(num_initial_improvement_steps):
        solution = improvement_step(solution, model)

    epoch_loss = 0.0
    epoch_reward = 0.0

    for _ in range(num_iterations):
        rollout = model.rollout(
            solution,
            rollouts_per_solution=rollouts_per_solution,
            deconstruction_size=deconstruction_size,
            greedy=False,
        )

        solution_partial = apply_rollout(solution, rollout)
        assert solution_partial.batch_dims == 2
        solution_prime = repair(solution_partial, rollout)

        reward = torch.clamp(
            solution.objective().unsqueeze(-1) - solution_prime.objective(),
            min=0.0,
        )
        baseline = reward.mean(dim=1, keepdim=True)
        best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)

        best_reward = reward.gather(1, best_rollout_idx)
        best_log_p = rollout.overall_log_p.gather(1, best_rollout_idx)
        loss = -((best_reward - baseline) * best_log_p).mean()
        loss.backward()

        epoch_loss += loss.item()
        epoch_reward += best_reward.mean().item()

        solution = solution_prime.gather(1, best_rollout_idx.squeeze()).squeeze(1)

    optimizer.step()

    return EpochSummary(
        epoch=epoch,
        loss=epoch_loss / num_iterations,
        reward=epoch_reward / num_iterations,
    )
