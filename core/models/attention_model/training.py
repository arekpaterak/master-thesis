import copy
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn as nn
from scipy import stats
from tqdm.auto import tqdm

from core.models.attention_model.model import AMRollout, AttentionModel
from core.problems.instance import BaseProblemInstance


class EvaluationResult(NamedTuple):
    lengths: torch.Tensor

    @property
    def mean(self) -> float:
        return self.lengths.mean().item()


class PairedTTestResult(NamedTuple):
    improved: bool
    obj_model_mean: float
    obj_baseline_mean: float


def evaluate(
    model: AttentionModel,
    instance_sampler: Callable[[int], BaseProblemInstance],
    n_instances: int = 10_000,
    batch_size: int = 512,
    device: str | torch.types.Device = "cpu",
) -> EvaluationResult:
    model.eval()
    all_lengths = []
    with torch.no_grad():
        for _ in range(0, n_instances, batch_size):
            bs = min(batch_size, n_instances - len(all_lengths) * batch_size)
            instances = instance_sampler(bs).to(device)
            rollout = model.rollout(instances, greedy=True)
            solution = rollout.to_solution(instances)
            all_lengths.append(solution.objective().cpu())
    return EvaluationResult(lengths=torch.cat(all_lengths))


def one_sided_paired_ttest(
    model: AttentionModel,
    baseline: AttentionModel,
    instance_sampler: Callable[[int], BaseProblemInstance],
    n_instances: int = 10_000,
    batch_size: int = 512,
    alpha: float = 0.05,
    device: str | torch.types.Device = "cpu",
) -> PairedTTestResult:
    """One-sided paired t-test: is model significantly better than baseline?"""
    obj_model = evaluate(model, instance_sampler, n_instances, batch_size, device).lengths
    obj_baseline = evaluate(baseline, instance_sampler, n_instances, batch_size, device).lengths
    t, p = stats.ttest_rel(obj_model.numpy(), obj_baseline.numpy())
    p_one_sided = p / 2 if t < 0 else 1.0
    return PairedTTestResult(
        improved=(p_one_sided < alpha),
        obj_model_mean=obj_model.mean().item(),
        obj_baseline_mean=obj_baseline.mean().item(),
    )


def freeze_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def sync_baseline_(baseline: AttentionModel, model: AttentionModel) -> None:
    baseline.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline.eval()
    freeze_parameters(baseline)


def prepare_baseline(model: AttentionModel) -> AttentionModel:
    baseline = copy.deepcopy(model)
    sync_baseline_(baseline, model)
    return baseline


class EpochSummary(NamedTuple):
    epoch: int
    loss: float
    advantage: float
    obj_model: float
    obj_baseline: float
    baseline_updated: bool
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float
    instances_per_sec: float | None = None


def train_epoch(
    *,
    epoch: int,
    model: AttentionModel,
    baseline: AttentionModel,
    optimizer: torch.optim.Optimizer,
    instance_sampler: Callable[[int], BaseProblemInstance],
    steps_per_epoch: int = 2500,
    batch_size: int = 512,
    alpha: float = 0.05,
    eval_n_instances: int = 10_000,
    eval_batch_size: int = 512,
    on_batch_end: Callable[[int, float], None] | None = None,
) -> EpochSummary:
    """
    Train one Attention Model epoch using REINFORCE with a greedy rollout baseline.
    """
    device = next(model.parameters()).device
    model.train()

    epoch_start = time.perf_counter()
    epoch_loss = 0.0
    epoch_entropy = 0.0
    epoch_entropy_norm = 0.0
    epoch_grad_norm = 0.0
    epoch_advantage = 0.0
    epoch_log_p = 0.0

    step_bar = tqdm(
        range(1, steps_per_epoch + 1),
        desc=f"Epoch {epoch}",
        unit="step",
        leave=False,
    )
    for step in step_bar:
        instances = instance_sampler(batch_size).to(device)

        rollout: AMRollout = model.rollout(instances, greedy=False)

        with torch.no_grad():
            baseline_rollout = baseline.rollout(instances, greedy=True)

        obj_model = rollout.to_solution(instances).objective()
        obj_baseline = baseline_rollout.to_solution(instances).objective()
        advantage = (obj_model - obj_baseline).detach()
        loss = (advantage * rollout.overall_log_p).mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_entropy += rollout.entropy or 0.0
        epoch_entropy_norm += rollout.entropy_norm or 0.0
        epoch_grad_norm += grad_norm.item()
        epoch_advantage += advantage.mean().item()
        epoch_log_p += rollout.log_p_of_actions.mean().item()

        step_bar.set_postfix(
            OrderedDict(
                loss=f"{loss.item():.4f}",
                entropy=f"{rollout.entropy:.3f}",
                entropy_norm=f"{rollout.entropy_norm:.3f}",
            )
        )

        if on_batch_end is not None:
            on_batch_end(step, loss.item())

    elapsed = time.perf_counter() - epoch_start
    instances_per_sec = (steps_per_epoch * batch_size) / elapsed

    test = one_sided_paired_ttest(
        model,
        baseline,
        instance_sampler,
        n_instances=eval_n_instances,
        batch_size=eval_batch_size,
        alpha=alpha,
        device=device,
    )
    if test.improved:
        sync_baseline_(baseline, model)

    return EpochSummary(
        epoch=epoch,
        loss=(epoch_loss / steps_per_epoch),
        advantage=(epoch_advantage / steps_per_epoch),
        obj_model=test.obj_model_mean,
        obj_baseline=test.obj_baseline_mean,
        baseline_updated=test.improved,
        entropy=(epoch_entropy / steps_per_epoch),
        entropy_norm=(epoch_entropy_norm / steps_per_epoch),
        grad_norm=(epoch_grad_norm / steps_per_epoch),
        log_p=(epoch_log_p / steps_per_epoch),
        instances_per_sec=instances_per_sec,
    )
