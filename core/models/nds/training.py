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
from core.models.nds.baseline import Baseline
from core.models.nds.factory import NDSModel
from core.models.nds.models.split import SplitNDSModel
from core.models.nds.policy_loss import PolicyLoss
from core.models.nds.reward import Reward
from core.models.nds.utils import diversity_overlap, unique_rollouts
from core.models.rollout import Rollout
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class EncoderPass:
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


class RecomputePass(EncoderPass):
    """Re-encode the instance every improvement step (rebuilds the graph each step)."""

    def __init__(self, model: NDSModel) -> None:
        self._model = model

    def rollout(self, solution: BaseSolution, **kwargs) -> Rollout:
        return self._model.rollout(solution, **kwargs)


class EncodeOncePass(EncoderPass):
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
        return self.objective_before - self.objective_after


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
    baseline: Baseline,
    repair: RepairOperator,
    n_repairs: int = 1,
    rollouts_per_solution: int = 1,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: torch.amp.GradScaler | None = None,
    encode_once: bool = False,
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

        objective_start = solution.objective().mean().item()

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
        total_reward_std = 0.0
        total_acceptance_rate = 0.0
        total_unique_actions_ratio = 0.0
        total_diversity_overlap = 0.0
        total_unique_rollouts = 0.0

        # Encode the static instance once and reuse it across improvement steps, or
        # recompute it every step. See EncodeOncePass / RecomputePass.
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            encoder_pass: EncoderPass = (
                EncodeOncePass(model, solution.instance) if encode_once else RecomputePass(model)
            )

        for _step in improvement_step_progress(range(num_main_improvement_steps)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                rollout = encoder_pass.rollout(
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
                baseline_reward = baseline(reward)
                reward_mean = reward.mean(dim=1, keepdim=True)
                best_rollout_idx = torch.argmax(reward, dim=1, keepdim=True)

                best_reward = reward.gather(1, best_rollout_idx)
                best_log_p = rollout.log_p.gather(1, best_rollout_idx)
                loss = policy_loss(reward=reward, baseline=baseline_reward, log_p=rollout.log_p)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            encoder_pass.after_backward()

            solution = solution_prime.gather(1, best_rollout_idx).squeeze(1)

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
                is_unique.float().sum(dim=-1).mean().item() / deconstruction_size
            )
            total_diversity_overlap += diversity_overlap(rollout.actions)
            total_unique_rollouts += unique_rollouts(rollout.actions)

        encoder_pass.finish()

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
            reward_std=total_reward_std / num_main_improvement_steps,
            acceptance_rate=total_acceptance_rate / num_main_improvement_steps,
            unique_actions_ratio=total_unique_actions_ratio / num_main_improvement_steps,
            diversity_overlap=total_diversity_overlap / num_main_improvement_steps,
            unique_rollouts=total_unique_rollouts / num_main_improvement_steps,
            objective_start=objective_start,
            objective_before=objective_before,
            objective_after=objective_after,
            grad_norm=grad_norm.item(),
            batch_time_sec=elapsed,
            instances_per_sec=batch_size / elapsed,
        )

    return training_step
