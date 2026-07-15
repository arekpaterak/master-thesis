from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Literal

import torch
import torch.optim as optim
from pydantic import BaseModel, computed_field

from core.lns.acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptance,
    UnconditionalAcceptance,
)
from core.lns.improvement_step import ImprovementStep
from core.lns.utils import select_solution
from core.models.module import EpochAccumulator
from core.models.nds.apply_and_repair import ApplyAndRepair
from core.models.nds.baseline import Baseline
from core.models.nds.factory import NDSModel
from core.models.nds.models.split import SplitNDSModel
from core.models.nds.policy_loss import PolicyLoss
from core.models.nds.reward import Reward
from core.models.nds.utils import diversity_overlap, unique_rollouts
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class ModelPass:
    """Strategy for producing per-improvement-step rollouts within a training step.

    Subclasses decide whether the instance encoder runs once or every step; the loop
    drives them uniformly: ``rollout`` each step, ``after_backward`` after every
    ``loss.backward``, and ``finish`` once after the loop.
    """

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        raise NotImplementedError

    def after_backward(self) -> None:
        pass

    def finish(self) -> None:
        pass


class RecomputePass(ModelPass):
    """Re-encode the instance every improvement step (rebuilds the graph each step)."""

    def __init__(self, model: NDSModel) -> None:
        self._model = model

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        return self._model.rollout(solution, **kwargs)


class EncodeOncePass(ModelPass):
    """Encode the instance once and reuse it across improvement steps.

    The encoder output is detached into a leaf so each step's backward stays bounded;
    its gradient is accumulated and pushed through the encoder once in ``finish``. This
    is exact: the split encoder is a deterministic function of the instance.
    """

    def __init__(self, model: SplitNDSModel, instance: BaseProblemInstance) -> None:
        self._model = model
        base_cache = model.encode_instance(instance)
        self._embeddings = base_cache.instance_node_embeddings
        self._embeddings_detached = self._embeddings.detach().requires_grad_(True)
        self._cache = base_cache._replace(instance_node_embeddings=self._embeddings_detached)
        self._accumulated_gradient = torch.zeros_like(self._embeddings)

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        return self._model.rollout(solution, cache=self._cache, **kwargs)

    def after_backward(self) -> None:
        self._accumulated_gradient += self._embeddings_detached.grad
        self._embeddings_detached.grad = None

    def finish(self) -> None:
        """One encoder backward with the summed boundary gradient."""
        self._embeddings.backward(gradient=self._accumulated_gradient)


class BatchTrainSummary(BaseModel):
    model_config = {"frozen": True}

    batch_size: int
    loss: float
    reward_best: float
    reward_mean: float
    entropy: float
    entropy_norm: float
    log_p: float
    reward_std: float
    acceptance_rate: float
    unique_actions_ratio: float
    diversity_overlap: float
    unique_rollouts: float
    objective_start: float
    objective_before: float
    objective_after: float
    grad_norm: float
    batch_time_sec: float
    instances_per_sec: float

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_start - self.objective_after

    @computed_field
    @property
    def relative_improvement(self) -> float:
        if self.objective_start == 0.0:
            return 0.0
        return self.improvement / self.objective_start


class EpochTrainSummary(BaseModel):
    model_config = {"frozen": True}

    epoch: int
    loss: float
    best_reward: float
    reward_mean: float
    entropy: float
    entropy_norm: float
    log_p: float
    reward_std: float
    acceptance_rate: float
    unique_actions_ratio: float
    diversity_overlap: float
    unique_rollouts: float
    objective_start: float
    objective_before: float
    objective_after: float
    grad_norm: float
    batch_time_sec: float
    epoch_time_sec: float
    instances_per_sec: float

    @computed_field
    @property
    def improvement(self) -> float:
        return self.objective_start - self.objective_after

    @computed_field
    @property
    def relative_improvement(self) -> float:
        if self.objective_start == 0.0:
            return 0.0
        return self.improvement / self.objective_start


class EpochTrainAccumulator(EpochAccumulator):
    def __init__(self) -> None:
        self._total_instances = 0
        self._total_batches = 0
        self._total_time = 0.0
        self._total_batch_time = 0.0
        self._total_loss = 0.0
        self._total_best_reward = 0.0
        self._total_reward_mean = 0.0
        self._total_entropy = 0.0
        self._total_entropy_norm = 0.0
        self._total_log_p = 0.0
        self._total_reward_std = 0.0
        self._total_acceptance_rate = 0.0
        self._total_unique_actions_ratio = 0.0
        self._total_diversity_overlap = 0.0
        self._total_unique_rollouts = 0.0
        self._total_objective_start = 0.0
        self._total_objective_before = 0.0
        self._total_objective_after = 0.0
        self._total_grad_norm = 0.0

    def update(self, summary: BatchTrainSummary) -> None:
        n = summary.batch_size
        self._total_instances += n
        self._total_batches += 1
        self._total_time += summary.batch_time_sec
        self._total_batch_time += summary.batch_time_sec
        self._total_loss += summary.loss * n
        self._total_best_reward += summary.reward_best * n
        self._total_reward_mean += summary.reward_mean * n
        self._total_entropy += summary.entropy * n
        self._total_entropy_norm += summary.entropy_norm * n
        self._total_log_p += summary.log_p * n
        self._total_reward_std += summary.reward_std * n
        self._total_acceptance_rate += summary.acceptance_rate * n
        self._total_unique_actions_ratio += summary.unique_actions_ratio * n
        self._total_diversity_overlap += summary.diversity_overlap * n
        self._total_unique_rollouts += summary.unique_rollouts * n
        self._total_objective_start += summary.objective_start * n
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
            reward_std=self._total_reward_std / n,
            acceptance_rate=self._total_acceptance_rate / n,
            unique_actions_ratio=self._total_unique_actions_ratio / n,
            diversity_overlap=self._total_diversity_overlap / n,
            unique_rollouts=self._total_unique_rollouts / n,
            objective_start=self._total_objective_start / n,
            objective_before=self._total_objective_before / n,
            objective_after=self._total_objective_after / n,
            grad_norm=self._total_grad_norm / n,
            batch_time_sec=self._total_batch_time / self._total_batches,
            epoch_time_sec=self._total_batch_time,
            instances_per_sec=self._total_instances / self._total_time,
        )


class NDSTrainingStep:
    def __init__(
        self,
        *,
        num_main_improvement_steps: int,
        num_initial_improvement_steps: int,
        deconstruction_size: int,
        generate_start_solution: Callable[[BaseProblemInstance], BaseSolution],
        improvement_step: ImprovementStep,
        apply_and_repair: ApplyAndRepair,
        compute_reward: Reward,
        policy_loss: PolicyLoss,
        baseline: Baseline,
        rollouts_per_solution: int = 1,
        amp_enabled: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        scaler: torch.amp.GradScaler | None = None,
        encode_once: bool = False,
        improvement_step_progress: Callable[[Iterable[int]], Iterable[int]] = lambda it: it,
        rollout_selection: Literal["reward", "objective"] = "objective",
        acceptance: AcceptanceCriterion = UnconditionalAcceptance(),
    ) -> None:
        self._num_main_improvement_steps = num_main_improvement_steps
        self._num_initial_improvement_steps = num_initial_improvement_steps
        self._deconstruction_size = deconstruction_size
        self._generate_start_solution = generate_start_solution
        self._improvement_step = improvement_step
        self._apply_and_repair = apply_and_repair
        self._compute_reward = compute_reward
        self._policy_loss = policy_loss
        self._baseline = baseline
        self._rollouts_per_solution = rollouts_per_solution
        self._amp_enabled = amp_enabled
        self._amp_dtype = amp_dtype
        self._scaler = scaler
        self._encode_once = encode_once
        self._improvement_step_progress = improvement_step_progress
        self._rollout_selection = rollout_selection
        self._acceptance = acceptance

    def __call__(
        self,
        model: NDSModel,
        optimizer: optim.Optimizer,
        instance: BaseProblemInstance,
    ) -> BatchTrainSummary:
        start = time.perf_counter()
        device = next(model.parameters()).device
        instance = instance.to(device)

        solution = self._generate_start_solution(instance)

        objective_start = solution.objective().mean().item()

        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ),
        ):
            for _ in range(self._num_initial_improvement_steps):
                solution = self._improvement_step(solution, GreedyAcceptance())

        model.train()
        optimizer.zero_grad()
        objective_before = solution.objective().mean().item()

        total_loss = 0.0
        total_reward_best = 0.0
        total_reward_mean = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_log_p = 0.0
        total_reward_std = 0.0
        total_acceptance_rate = 0.0
        total_unique_actions_ratio = 0.0
        total_diversity_overlap = 0.0
        total_unique_rollouts = 0.0

        # Encode the static instance once and reuse it across improvement steps, or
        # recompute it every step. See EncodeOncePass / RecomputePass.
        with torch.autocast(
            device_type=device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
        ):
            encoder_pass: ModelPass = (
                EncodeOncePass(model, solution.instance)
                if self._encode_once
                else RecomputePass(model)
            )

        for _step in self._improvement_step_progress(range(self._num_main_improvement_steps)):
            with torch.autocast(
                device_type=device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ):
                rollout = encoder_pass.rollout(
                    solution,
                    rollouts_per_solution=self._rollouts_per_solution,
                    deconstruction_size=self._deconstruction_size,
                    greedy=False,
                )

                solution_prime = self._apply_and_repair(solution, rollout)
                assert solution_prime.batch_dims == 2

                step_objective_before = solution.objective()
                objective_after = solution_prime.objective()
                reward = self._compute_reward(
                    objective_before=step_objective_before,
                    objective_after=objective_after,
                )
                baseline_reward = self._baseline(reward)
                reward_mean = reward.mean(dim=1, keepdim=True)

                if self._rollout_selection == "reward":
                    best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)
                else:
                    best_rollout_idx = torch.argmin(objective_after, dim=1, keepdim=True)

                best_reward = reward.gather(1, best_rollout_idx)
                best_log_p = rollout.log_p.gather(1, best_rollout_idx)
                loss = self._policy_loss(
                    reward=reward, baseline=baseline_reward, log_p=rollout.log_p
                )

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            encoder_pass.after_backward()

            best_solution_prime = solution_prime.gather(1, best_rollout_idx).squeeze(1)
            mask = self._acceptance(solution, best_solution_prime)
            solution = select_solution(mask, best_solution_prime, solution)

            sorted_actions = rollout.actions.sort(dim=-1).values
            is_unique = torch.cat(
                [
                    torch.ones(
                        *sorted_actions.shape[:-1],
                        1,
                        dtype=torch.bool,
                        device=sorted_actions.device,
                    ),
                    sorted_actions[..., 1:] != sorted_actions[..., :-1],
                ],
                dim=-1,
            )

            total_loss += loss.item()
            total_reward_best += best_reward.mean().item()
            total_reward_mean += reward_mean.mean().item()
            total_entropy += rollout.mean_entropy
            total_entropy_norm += rollout.mean_entropy_norm
            total_log_p += best_log_p.mean().item()
            total_reward_std += reward.std(dim=1).mean().item()
            total_acceptance_rate += (best_reward > 0).float().mean().item()
            total_unique_actions_ratio += (
                is_unique.float().sum(dim=-1).mean().item() / self._deconstruction_size
            )
            total_diversity_overlap += diversity_overlap(rollout.actions)
            total_unique_rollouts += unique_rollouts(rollout.actions)

        encoder_pass.finish()

        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

        objective_after = solution.objective().mean().item()

        elapsed = time.perf_counter() - start
        batch_size = instance.batch_size[0]

        return BatchTrainSummary(
            batch_size=batch_size,
            loss=total_loss / self._num_main_improvement_steps,
            reward_best=total_reward_best / self._num_main_improvement_steps,
            reward_mean=total_reward_mean / self._num_main_improvement_steps,
            entropy=total_entropy / self._num_main_improvement_steps,
            entropy_norm=total_entropy_norm / self._num_main_improvement_steps,
            log_p=total_log_p / self._num_main_improvement_steps,
            reward_std=total_reward_std / self._num_main_improvement_steps,
            acceptance_rate=total_acceptance_rate / self._num_main_improvement_steps,
            unique_actions_ratio=total_unique_actions_ratio / self._num_main_improvement_steps,
            diversity_overlap=total_diversity_overlap / self._num_main_improvement_steps,
            unique_rollouts=total_unique_rollouts / self._num_main_improvement_steps,
            objective_start=objective_start,
            objective_before=objective_before,
            objective_after=objective_after,
            grad_norm=grad_norm.item(),
            batch_time_sec=elapsed,
            instances_per_sec=batch_size / elapsed,
        )
