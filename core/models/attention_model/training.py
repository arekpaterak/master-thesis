import copy
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from scipy import stats
from tqdm.auto import tqdm

from core.models.attention_model.model import AttentionModel, Rollout
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


def freeze_parameters_(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


@dataclass(frozen=True)
class BatchSummary:
    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    loss: float
    advantage: float
    entropy: float
    entropy_norm: float
    grad_norm: float
    log_p: float
    batches: int
    batch_size: int
    obj_model: float
    obj_baseline: float
    baseline_updated: bool
    elapsed_seconds: float

    @property
    def instances(self) -> int:
        return self.batches * self.batch_size

    @property
    def instances_per_sec(self) -> float:
        return self.instances / self.elapsed_seconds

    def asdict(self) -> dict[str, float | int | bool]:
        data = asdict(self)
        data["instances_per_sec"] = self.instances_per_sec
        return data


def _training_step(
    *,
    model: AttentionModel,
    baseline: AttentionModel,
    optimizer: torch.optim.Optimizer,
    instance: BaseProblemInstance,
) -> BatchSummary:
    rollout: Rollout = model.rollout(instance, greedy=False)

    with torch.no_grad():
        baseline_rollout = baseline.rollout(instance, greedy=True)

    obj_model = rollout.to_solution(instance).objective()
    obj_baseline = baseline_rollout.to_solution(instance).objective()
    advantage = (obj_model - obj_baseline).detach()
    loss = (advantage * rollout.overall_log_p).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return BatchSummary(
        loss=loss.item(),
        advantage=advantage.mean().item(),
        entropy=rollout.entropy or 0.0,
        entropy_norm=rollout.entropy_norm or 0.0,
        grad_norm=grad_norm.item(),
        log_p=rollout.log_p_of_actions.mean().item(),
    )


def configure_train_epoch(
    *,
    instance_sampler: Callable[[int], BaseProblemInstance],
    batches_per_epoch: int = 2500,
    batch_size: int = 512,
    alpha: float = 0.05,
    eval_n_instances: int = 10_000,
    eval_batch_size: int = 512,
    on_batch_end: Callable[[int, float], None] | None = None,
) -> Callable[..., EpochSummary]:
    def train_epoch(
        *,
        epoch: int,
        model: AttentionModel,
        optimizer: torch.optim.Optimizer,
        baseline: AttentionModel,
    ) -> EpochSummary:
        device = next(model.parameters()).device
        model.train()

        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_entropy = 0.0
        total_entropy_norm = 0.0
        total_grad_norm = 0.0
        total_advantage = 0.0
        total_log_p = 0.0

        step_bar = tqdm(
            range(1, batches_per_epoch + 1),
            desc=f"Epoch {epoch}",
            unit="batch",
            leave=False,
        )
        for batch_idx in step_bar:
            instance = instance_sampler(batch_size).to(device)
            batch_summary = _training_step(
                model=model,
                baseline=baseline,
                optimizer=optimizer,
                instance=instance,
            )

            total_loss += batch_summary.loss
            total_entropy += batch_summary.entropy
            total_entropy_norm += batch_summary.entropy_norm
            total_grad_norm += batch_summary.grad_norm
            total_advantage += batch_summary.advantage
            total_log_p += batch_summary.log_p

            step_bar.set_postfix(
                OrderedDict(
                    loss=f"{batch_summary.loss:.4f}",
                    entropy=f"{batch_summary.entropy:.3f}",
                    entropy_norm=f"{batch_summary.entropy_norm:.3f}",
                )
            )

            if on_batch_end is not None:
                on_batch_end(batch_idx, batch_summary.loss)

        elapsed = time.perf_counter() - epoch_start

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
            loss=total_loss / batches_per_epoch,
            advantage=total_advantage / batches_per_epoch,
            entropy=total_entropy / batches_per_epoch,
            entropy_norm=total_entropy_norm / batches_per_epoch,
            grad_norm=total_grad_norm / batches_per_epoch,
            log_p=total_log_p / batches_per_epoch,
            batches=batches_per_epoch,
            batch_size=batch_size,
            obj_model=test.obj_model_mean,
            obj_baseline=test.obj_baseline_mean,
            baseline_updated=test.improved,
            elapsed_seconds=elapsed,
        )

    return train_epoch


def sync_baseline_(baseline: AttentionModel, model: AttentionModel) -> None:
    baseline.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline.eval()
    freeze_parameters_(baseline)


def prepare_baseline(model: AttentionModel) -> AttentionModel:
    baseline = copy.deepcopy(model)
    sync_baseline_(baseline, model)
    return baseline
