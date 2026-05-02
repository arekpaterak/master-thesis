import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.optim as optim

from core.models.nds.model import NDSModel
from core.models.nds.rollout import NDSRollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@dataclass(frozen=True)
class BatchSummary:
    loss: float
    reward: float
    reward_baseline: float
    entropy: float
    entropy_norm: float
    log_p: float
    objective_before: float
    objective_after: float
    grad_norm: float


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    total_loss: float
    total_reward: float
    total_reward_baseline: float
    total_entropy: float
    total_entropy_norm: float
    total_log_p: float
    objective_before: float
    objective_after: float
    total_grad_norm: float
    batches: int
    iterations: int
    elapsed_seconds: float

    @property
    def total_iterations(self) -> int:
        return self.batches * self.iterations

    @property
    def loss(self) -> float:
        return self.total_loss / self.total_iterations

    @property
    def reward(self) -> float:
        return self.total_reward / self.total_iterations

    @property
    def reward_baseline(self) -> float:
        return self.total_reward_baseline / self.total_iterations

    @property
    def entropy(self) -> float:
        return self.total_entropy / self.total_iterations

    @property
    def entropy_norm(self) -> float:
        return self.total_entropy_norm / self.total_iterations

    @property
    def log_p(self) -> float:
        return self.total_log_p / self.total_iterations

    @property
    def improvement(self) -> float:
        return self.objective_before - self.objective_after

    @property
    def grad_norm(self) -> float:
        return self.total_grad_norm / self.batches

    @property
    def iterations_per_sec(self) -> float:
        return self.total_iterations / self.elapsed_seconds

    def asdict(self) -> dict[str, float | int]:
        data = asdict(self)
        data.update(
            {
                "loss": self.loss,
                "reward": self.reward,
                "reward_baseline": self.reward_baseline,
                "total_iterations": self.total_iterations,
                "entropy": self.entropy,
                "entropy_norm": self.entropy_norm,
                "log_p": self.log_p,
                "objective_before": self.objective_before,
                "objective_after": self.objective_after,
                "improvement": self.improvement,
                "grad_norm": self.grad_norm,
                "iterations_per_sec": self.iterations_per_sec,
            }
        )
        return data


def _train_batch(
    *,
    model: NDSModel,
    optimizer: optim.Optimizer,
    instance: BaseProblemInstance,
    num_main_improvement_steps: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution],
    apply_rollout: Callable[[BaseSolution, NDSRollout], BaseSolution],
    repair: Callable[[BaseSolution, NDSRollout], BaseSolution],
) -> BatchSummary:
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

        total_loss += loss.item()
        total_reward += best_reward.mean().item()
        total_reward_baseline += reward_baseline.mean().item()
        total_entropy += rollout.entropy or 0.0
        total_entropy_norm += rollout.entropy_norm or 0.0
        total_log_p += best_log_p.mean().item()

        solution = solution_prime.gather(1, best_rollout_idx.squeeze()).squeeze(1)

    objective_after = solution.objective().mean().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

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
    )


def configure_train_epoch(
    *,
    instance_sampler: Callable[[int], BaseProblemInstance],
    batches_per_epoch: int,
    batch_size: int,
    num_main_improvement_steps: int,
    rollouts_per_solution: int,
    num_initial_improvement_steps: int,
    deconstruction_size: int,
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
    improvement_step: Callable[[BaseSolution, NDSModel], BaseSolution],
    apply_rollout: Callable[[BaseSolution, NDSRollout], BaseSolution],
    repair: Callable[[BaseSolution, NDSRollout], BaseSolution],
) -> Callable[..., EpochSummary]:
    def train_epoch(
        *,
        epoch: int,
        model: NDSModel,
        optimizer: optim.Optimizer,
    ) -> EpochSummary:
        model.train()

        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_reward = 0.0
        total_reward_baseline = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_log_p = 0.0
        total_objective_before = 0.0
        total_objective_after = 0.0
        total_grad_norm = 0.0

        for _ in range(batches_per_epoch):
            instance = instance_sampler(batch_size)
            batch_summary = _train_batch(
                model=model,
                optimizer=optimizer,
                instance=instance,
                num_main_improvement_steps=num_main_improvement_steps,
                rollouts_per_solution=rollouts_per_solution,
                num_initial_improvement_steps=num_initial_improvement_steps,
                deconstruction_size=deconstruction_size,
                generate_start_solution=generate_start_solution,
                improvement_step=improvement_step,
                apply_rollout=apply_rollout,
                repair=repair,
            )

            total_loss += batch_summary.loss
            total_reward += batch_summary.reward
            total_reward_baseline += batch_summary.reward_baseline
            total_entropy += batch_summary.entropy
            total_entropy_norm += batch_summary.entropy_norm
            total_log_p += batch_summary.log_p
            total_objective_before += batch_summary.objective_before
            total_objective_after += batch_summary.objective_after
            total_grad_norm += batch_summary.grad_norm

        return EpochSummary(
            epoch=epoch,
            total_loss=total_loss,
            total_reward=total_reward,
            total_reward_baseline=total_reward_baseline,
            total_entropy=total_entropy,
            total_entropy_norm=total_entropy_norm,
            total_log_p=total_log_p,
            objective_before=total_objective_before / batches_per_epoch,
            objective_after=total_objective_after / batches_per_epoch,
            total_grad_norm=total_grad_norm,
            batches=batches_per_epoch,
            iterations=num_main_improvement_steps,
            elapsed_seconds=time.perf_counter() - epoch_start,
        )

    return train_epoch
