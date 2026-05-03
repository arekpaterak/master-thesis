from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.optim as optim
from pydantic import BaseModel, computed_field

from core.models.nds.model import NDSModel
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class BatchSummary(BaseModel):
    model_config = {"frozen": True}

    loss: float
    reward: float
    reward_baseline: float
    entropy: float
    entropy_norm: float
    log_p: float
    objective_before: float
    objective_after: float
    grad_norm: float
    instances_per_sec: float


class EpochSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    reward: float
    reward_baseline: float
    entropy: float
    entropy_norm: float
    log_p: float
    objective_before: float
    objective_after: float
    grad_norm: float
    instances_per_sec: float

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_before - self.objective_after

    @classmethod
    def from_step_summaries(
        cls,
        *,
        epoch: int,
        batch_summaries: list[BatchSummary],
        **_,
    ) -> EpochSummary:
        n = len(batch_summaries)
        return cls(
            epoch=epoch,
            loss=sum(s.loss for s in batch_summaries) / n,
            reward=sum(s.reward for s in batch_summaries) / n,
            reward_baseline=sum(s.reward_baseline for s in batch_summaries) / n,
            entropy=sum(s.entropy for s in batch_summaries) / n,
            entropy_norm=sum(s.entropy_norm for s in batch_summaries) / n,
            log_p=sum(s.log_p for s in batch_summaries) / n,
            objective_before=sum(s.objective_before for s in batch_summaries) / n,
            objective_after=sum(s.objective_after for s in batch_summaries) / n,
            grad_norm=sum(s.grad_norm for s in batch_summaries) / n,
            instances_per_sec=sum(s.instances_per_sec for s in batch_summaries) / n,
        )


def configure_training_step(
    *,
    num_main_improvement_steps: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution],
    apply_rollout: Callable[[BaseSolution, Rollout], BaseSolution],
    repair: Callable[[BaseSolution, Rollout], BaseSolution],
) -> Callable[..., BatchSummary]:
    def training_step(
        *,
        model: NDSModel,
        optimizer: optim.Optimizer,
        instance: BaseProblemInstance,
        **_,
    ) -> BatchSummary:
        start = time.perf_counter()
        device = next(model.parameters()).device
        instance = instance.to(device)
        model.train()

        # TODO: For future, think if only operate on one instance, expanded to many rollouts, instead of batch of instances
        # Like in the article
        optimizer.zero_grad()
        solution = generate_start_solution(instance)

        for _ in range(num_initial_improvement_steps):
            solution = improvement_step(solution, model)

        objective_before = solution.objective().mean().item()

        total_loss = 0.0
        total_reward = 0.0
        total_reward_baseline = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_log_p = 0.0

        for _ in range(num_main_improvement_steps):
            rollout = model.rollout(
                solution,
                rollouts_per_solution=rollouts_per_solution,
                deconstruction_size=deconstruction_size,
                greedy=False,
            )

            solution_partial = apply_rollout(solution, rollout)
            assert solution_partial.batch_dims == 2
            solution_prime = repair(solution_partial, rollout)

            step_objective_before = solution.objective()
            objective_after = solution_prime.objective()
            reward = torch.clamp(
                step_objective_before.unsqueeze(-1) - objective_after,
                min=0.0,
            )
            reward_baseline = reward.mean(dim=1, keepdim=True)
            best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)

            best_reward = reward.gather(1, best_rollout_idx)
            best_log_p = rollout.overall_log_p.gather(1, best_rollout_idx)
            loss = -((best_reward - reward_baseline) * best_log_p).mean()
            loss.backward()

            solution = solution_prime.gather(1, best_rollout_idx.squeeze()).squeeze(1)

            total_loss += loss.item()
            total_reward += best_reward.mean().item()
            total_reward_baseline += reward_baseline.mean().item()
            total_entropy += rollout.entropy or 0.0
            total_entropy_norm += rollout.entropy_norm or 0.0
            total_log_p += best_log_p.mean().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        objective_after = solution.objective().mean().item()

        elapsed = time.perf_counter() - start
        batch_size = instance.batch_size[0]

        # TODO: Normalize by main improvement steps
        return BatchSummary(
            loss=total_loss,
            reward=total_reward,
            reward_baseline=total_reward_baseline,
            entropy=total_entropy,
            entropy_norm=total_entropy_norm,
            log_p=total_log_p,
            objective_before=objective_before,
            objective_after=objective_after,
            grad_norm=grad_norm.item(),
            instances_per_sec=batch_size / elapsed,
        )

    return training_step
