from collections.abc import Callable

import torch
import torch.optim as optim

from core.models.nds.model import NDSModel
from core.models.nds.rollout import NDSRollout, rollout_policy
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


def train(
    model: NDSModel,
    optimizer: optim.Optimizer,
    *,
    num_iterations: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    max_epochs: int,
    generate_instance: Callable[[], BaseProblemInstance],
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution],
    apply_rollout: Callable[[BaseSolution, NDSRollout], BaseSolution],
    repair: Callable[[BaseSolution, NDSRollout], BaseSolution],
):
    """
    Args:
        model:
        optimizer:
        num_iterations: iterations per instance
        rollouts_per_solution: rollouts per solution
        num_initial_improvement_steps: initial improvement steps
        deconstruction_size: deconstruction size
        max_epochs: maximum number of epochs
        generate_instance:
        generate_start_solution:
        improvement_step:
        apply_rollout:
        repair:

    Returns:

    """
    model.train()

    for epoch in range(max_epochs):
        # 4: Generate problem instance
        instance = generate_instance()

        # 5: Generate initial solution
        solution = generate_start_solution(instance)

        # 6-8: Initial improvement steps
        for j in range(num_initial_improvement_steps):
            solution = improvement_step(solution, model)

        # 9: Main iterations
        for i in range(num_iterations):
            # 10: Sample K rollouts (keep log_probs for backprop)
            rollout = rollout_policy(
                model,
                solution,
                rollouts_per_solution=rollouts_per_solution,
                deconstruction_size=deconstruction_size,
                greedy=False,
            )

            # 11-12: Deconstruct and reconstruct
            solution_partial = apply_rollout(solution, rollout)  # (B, R)
            solution_prime = repair(solution_partial, rollout)  # (B, R)

            # 13: Calculate reward
            reward = torch.clamp(
                solution.objective().unsqueeze(-1) - solution_prime.objective(),
                min=0.0,
            )

            # 14: Calculate baseline
            b = reward.mean(
                dim=1, keepdim=True
            )  # (B, 1) average over rollouts for each problem instance in the batch

            # 15: Find the best rollout in each batch
            best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)  # (B, 1)

            # 16: Calculate gradients for the best rollout
            log_p = rollout.log_p_of_actions
            best_reward = reward.gather(1, best_rollout_idx)
            best_log_p = log_p.gather(1, best_rollout_idx)
            loss_i = -((best_reward - b) * best_log_p).mean()
            loss_i.backward()

            # 17: Update the current solution with best found solution
            solution = solution_prime.gather(1, best_rollout_idx.squeeze()).squeeze(1)  # (B,)

        # 19: Optimizer step (with gradients accumulated over iterations)
        optimizer.step()
        optimizer.zero_grad()
