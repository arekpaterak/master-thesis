from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Literal

import torch
import torch.optim as optim

from core.lns.acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptance,
    UnconditionalAcceptance,
)
from core.lns.apply_and_repair import ApplyAndRepair
from core.lns.improvement_step import ImprovementStep
from core.lns.operators.destroy.selection import NodeSelection
from core.models.nds.baseline import Baseline
from core.models.nds.metrics import EncoderMetricsRecorder, diversity_overlap, unique_rollouts
from core.models.nds.models import NDSModel
from core.models.nds.policy_loss import PolicyLoss
from core.models.nds.reward import Reward
from core.models.nds.training.logs import BatchTrainSummary
from core.models.nds.training.passes import EncodeOncePass, ModelPass, RecomputePass
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution
from core.random import SeededGenerator


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
        seed: int | None = None,
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
        self._rng = SeededGenerator(seed)

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
            self._improvement_step.search_session(solution),
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
        total_improving_move_rate = 0.0
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

        with EncoderMetricsRecorder.attach(
            model.encoder, offset=model.components.num_context_nodes
        ) as encoder_recorder:
            for _step in self._improvement_step_progress(range(self._num_main_improvement_steps)):
                with torch.autocast(
                    device_type=device.type, dtype=self._amp_dtype, enabled=self._amp_enabled
                ):
                    rollout = encoder_pass.rollout(
                        solution,
                        rollouts_per_solution=self._rollouts_per_solution,
                        deconstruction_size=self._deconstruction_size,
                        rng=self._rng,
                        greedy=False,
                    )

                    solution_prime = self._apply_and_repair.parallel(
                        solution, NodeSelection(rollout.actions)
                    )
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
                solution = best_solution_prime.where(mask, solution)

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
                total_improving_move_rate += (best_reward > 0).float().mean().item()
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
            improving_move_rate=total_improving_move_rate / self._num_main_improvement_steps,
            unique_actions_ratio=total_unique_actions_ratio / self._num_main_improvement_steps,
            diversity_overlap=total_diversity_overlap / self._num_main_improvement_steps,
            unique_rollouts=total_unique_rollouts / self._num_main_improvement_steps,
            embeddings_cos_sim=encoder_recorder.embeddings_cos_sim,
            objective_start=objective_start,
            objective_before=objective_before,
            objective_after=objective_after,
            grad_norm=grad_norm.item(),
            batch_time_sec=elapsed,
            instances_per_sec=batch_size / elapsed,
        )
