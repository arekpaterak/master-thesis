from collections.abc import Callable

import torch
import torch.optim as optim
from pydantic import BaseModel, computed_field

from core.models.nds.model import NDSModel
from core.models.nds.rollout import NDSRollout
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

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_before - self.objective_after


def _training_step(
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
    # TODO: Think if only operate on one instance, expanded to many rollouts, instead of batch of instances
    optimizer.zero_grad()
    solution = generate_start_solution(instance)

    for _ in range(num_initial_improvement_steps):
        solution = improvement_step(solution, model)

    objective_before = solution.objective().mean().item()

    # Metrics across steps
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

    # TODO: Think if average metrics over number of improvement steps
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
    on_step_end: Callable[[BatchSummary], None] | None = None,
) -> Callable[..., EpochSummary]:
    def train_epoch(
        *,
        epoch: int,
        model: NDSModel,
        optimizer: optim.Optimizer,
    ) -> EpochSummary:
        device = next(model.parameters()).device
        model.train()

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
            instance = instance_sampler(batch_size).to(device)
            batch_summary = _training_step(
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

            if on_step_end is not None:
                on_step_end(batch_summary)

        return EpochSummary(
            epoch=epoch,
            loss=total_loss / batches_per_epoch,
            reward=total_reward / batches_per_epoch,
            reward_baseline=total_reward_baseline / batches_per_epoch,
            entropy=total_entropy / batches_per_epoch,
            entropy_norm=total_entropy_norm / batches_per_epoch,
            log_p=total_log_p / batches_per_epoch,
            objective_before=total_objective_before / batches_per_epoch,
            objective_after=total_objective_after / batches_per_epoch,
            grad_norm=total_grad_norm / batches_per_epoch,
        )

    return train_epoch
