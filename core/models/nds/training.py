from collections.abc import Callable

import torch
import torch.optim as optim

from core.models.nds.model import NDSModel
from core.models.nds.rollout import Rollout, rollout_policy
from core.problems.cvrp import CVRPInstance, CVRPSolution


def train(
    model: NDSModel,
    optimizer: optim.Optimizer,
    *,
    num_iterations: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    max_epochs: int,
    generate_instance: Callable[[], CVRPInstance],
    generate_start_solution: Callable[[CVRPInstance], CVRPSolution],
    improvement_step: Callable[[CVRPSolution, NDSModel], CVRPSolution],
    remove_customers: Callable[[CVRPSolution, Rollout], CVRPSolution],
    greedy_insertion: Callable[[CVRPSolution, Rollout], CVRPSolution],
    objective: Callable[[CVRPSolution], torch.Tensor],
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
        remove_customers:
        greedy_insertion:
        objective:

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
        for i in range(1, num_iterations + 1):
            # 10: Sample K rollouts (keep log_probs for backprop)
            rollout = rollout_policy(
                model,
                solution,
                rollouts_per_solution=rollouts_per_solution,
                deconstruction_size=deconstruction_size,
                greedy=False,
            )

            # 11-12: Deconstruct and reconstruct
            solution_partial = remove_customers(solution, rollout)
            solution_prime = greedy_insertion(solution_partial, rollout)

            # 13: Calculate reward
            reward = max(
                objective(solution) - objective(solution_prime),
                torch.zeros_like(objective(solution)),
            )

            # 14: Calculate baseline
            b = reward.mean()

            # 15: Find the best rollout in this iteration
            k_star = torch.argmax(reward)

            # 16: Calculate gradients for the best rollout
            log_p = rollout.log_p_of_actions
            loss_i = -(reward[k_star] - b) * log_p[k_star].sum()
            loss_i.backward()

            # 17: Update the current solution with best found solution
            solution = solution_prime[k_star]

        # 19: Optimizer step (with gradients accumulated over I iterations)
        optimizer.step()
        optimizer.zero_grad()

    return model
