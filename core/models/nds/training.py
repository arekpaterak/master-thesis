from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import torch
import torch.optim as optim
from pydantic import BaseModel, computed_field

from core.lns.acceptance_criterion import GreedyAcceptance
from core.lns.improvement_step import ImprovementStep
from core.lns.operators.repair import RepairOperator
from core.lns.utils import best_repair
from core.models.module import EpochAccumulator
from core.models.nds.apply_rollout import apply_rollout as default_apply_rollout
from core.models.nds.factory import NDSModel
from core.models.nds.policy_loss import PolicyLoss
from core.models.nds.reward import Reward
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class BatchTrainSummary(BaseModel):
    model_config = {"frozen": True}

    batch_size: int
    loss: float
    reward_best: float
    reward_mean: float
    entropy: float
    entropy_norm: float
    log_p: float
    objective_before: float
    objective_after: float
    grad_norm: float
    instances_per_sec: float


class EpochTrainSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    best_reward: float
    reward_mean: float
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


class EpochTrainAccumulator(EpochAccumulator):
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_time = 0.0
        self._total_loss = 0.0
        self._total_best_reward = 0.0
        self._total_reward_mean = 0.0
        self._total_entropy = 0.0
        self._total_entropy_norm = 0.0
        self._total_log_p = 0.0
        self._total_objective_before = 0.0
        self._total_objective_after = 0.0
        self._total_grad_norm = 0.0

    def update(self, summary: BatchTrainSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_time += n / summary.instances_per_sec
        self._total_loss += summary.loss * n
        self._total_best_reward += summary.reward_best * n
        self._total_reward_mean += summary.reward_mean * n
        self._total_entropy += summary.entropy * n
        self._total_entropy_norm += summary.entropy_norm * n
        self._total_log_p += summary.log_p * n
        self._total_objective_before += summary.objective_before * n
        self._total_objective_after += summary.objective_after * n
        self._total_grad_norm += summary.grad_norm * n

    def complete(self, *, epoch: int) -> EpochTrainSummary:
        n = self._total_instances
        return EpochTrainSummary(
            epoch=epoch,
            loss=self._total_loss / n,
            best_reward=self._total_best_reward / n,
            reward_mean=self._total_reward_mean / n,
            entropy=self._total_entropy / n,
            entropy_norm=self._total_entropy_norm / n,
            log_p=self._total_log_p / n,
            objective_before=self._total_objective_before / n,
            objective_after=self._total_objective_after / n,
            grad_norm=self._total_grad_norm / n,
            instances_per_sec=self._total_instances / self._total_time,
        )


def configure_training_step(
    *,
    num_main_improvement_steps: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: ImprovementStep,
    apply_rollout: Callable[[BaseSolution, Rollout], BaseSolution] | None = None,
    compute_reward: Reward,
    policy_loss: PolicyLoss,
    repair: RepairOperator,
    n_repairs: int = 1,
    rollouts_per_solution: int = 1,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: torch.amp.GradScaler | None = None,
    improvement_step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
) -> Callable[[NDSModel, optim.Optimizer, BaseProblemInstance], BatchTrainSummary]:
    _apply_rollout = default_apply_rollout if apply_rollout is None else apply_rollout

    def training_step(
        model: NDSModel,
        optimizer: optim.Optimizer,
        instance: BaseProblemInstance,
    ) -> BatchTrainSummary:
        start = time.perf_counter()
        device = next(model.parameters()).device
        instance = instance.to(device)

        solution = generate_start_solution(instance)

        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled),
        ):
            for _ in range(num_initial_improvement_steps):
                solution = improvement_step(solution, GreedyAcceptance())

        model.train()
        optimizer.zero_grad()
        objective_before = solution.objective().mean().item()

        total_loss = 0.0
        total_reward_best = 0.0
        total_reward_mean = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_log_p = 0.0

        for _step in improvement_step_progress(range(num_main_improvement_steps)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                rollout = model.rollout(
                    solution,
                    rollouts_per_solution=rollouts_per_solution,
                    deconstruction_size=deconstruction_size,
                    greedy=False,
                )

                solution_deconstructed = _apply_rollout(solution, rollout)
                assert solution_deconstructed.batch_dims == 2
                solution_prime = best_repair(repair, solution_deconstructed, n_repairs)

                step_objective_before = solution.objective()
                objective_after = solution_prime.objective()
                reward = compute_reward(
                    objective_before=step_objective_before,
                    objective_after=objective_after,
                )
                reward_mean = reward.mean(dim=1, keepdim=True)
                best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)

                best_reward = reward.gather(1, best_rollout_idx)
                best_log_p = rollout.log_p.gather(1, best_rollout_idx)
                loss = policy_loss(reward=reward, log_p=rollout.log_p)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            solution = solution_prime.gather(1, best_rollout_idx).squeeze(1)

            total_loss += loss.item()
            total_reward_best += best_reward.mean().item()
            total_reward_mean += reward_mean.mean().item()
            total_entropy += rollout.mean_entropy
            total_entropy_norm += rollout.mean_entropy_norm
            total_log_p += best_log_p.mean().item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        objective_after = solution.objective().mean().item()

        elapsed = time.perf_counter() - start
        batch_size = instance.batch_size[0]

        return BatchTrainSummary(
            batch_size=batch_size,
            loss=total_loss / num_main_improvement_steps,
            reward_best=total_reward_best / num_main_improvement_steps,
            reward_mean=total_reward_mean / num_main_improvement_steps,
            entropy=total_entropy / num_main_improvement_steps,
            entropy_norm=total_entropy_norm / num_main_improvement_steps,
            log_p=total_log_p / num_main_improvement_steps,
            objective_before=objective_before,
            objective_after=objective_after,
            grad_norm=grad_norm.item(),
            instances_per_sec=batch_size / elapsed,
        )

    return training_step
