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
            rollout = model(instances, greedy=True)
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
    # one-sided: H1 = model has lower (better) cost
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


def prepare_baseline(model: AttentionModel) -> AttentionModel:
    baseline = copy.deepcopy(model)
    baseline.eval()
    freeze_parameters(baseline)
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


def train(
    model: AttentionModel,
    instance_sampler: Callable[[int], BaseProblemInstance],
    n_epochs: int = 100,
    steps_per_epoch: int = 2500,
    batch_size: int = 512,
    lr: float = 1e-4,
    alpha: float = 0.05,
) -> list[EpochSummary]:
    """
    Train the Attention Model using REINFORCE with a greedy rollout baseline,
    as described in Algorithm 1 "Attention, Learn to Solve Routing Problems!" (Kool et al., 2019).

    Returns:
        (model, epoch_summaries)
    """
    device = next(model.parameters()).device

    # (Algorithm 1, line 2)
    baseline = prepare_baseline(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    epoch_summaries: list[EpochSummary] = []

    # (Algorithm 1, line 3)
    epoch_bar = tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()

        epoch_start = time.perf_counter()

        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_entropy_norm = 0.0
        epoch_grad_norm = 0.0
        epoch_advantage = 0.0
        epoch_log_p = 0.0

        # (Algorithm 1, line 4)
        step_bar = tqdm(
            range(1, steps_per_epoch + 1),
            desc=f"Epoch {epoch}",
            unit="step",
            leave=False,
        )
        for _ in step_bar:
            # Sample batch of random instances (Algorithm 1, line 5)
            instances = instance_sampler(batch_size).to(device)

            # Sample rollout from training policy (line 6)
            rollout: AMRollout = model(instances, greedy=False)

            # Greedy rollout from baseline policy (line 7)
            with torch.no_grad():
                baseline_rollout = baseline(instances, greedy=True)

            # REINFORCE gradient (line 8)
            L = rollout.to_solution(instances).objective()  # (batch,)
            L_baseline = baseline_rollout.to_solution(instances).objective()  # (batch,)

            advantage = (L - L_baseline).detach()  # (batch,)
            loss = (advantage * rollout.log_p.sum(dim=1)).mean()

            # (line 9)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_entropy += rollout.entropy
            epoch_entropy_norm += rollout.entropy_norm
            epoch_grad_norm += grad_norm.item()
            epoch_advantage += advantage.mean().item()
            epoch_log_p += rollout.log_p.mean().item()

            step_bar.set_postfix(
                OrderedDict(
                    loss=f"{loss.item():.4f}",
                    entropy=f"{rollout.entropy:.3f}",
                    entropy_norm=f"{rollout.entropy_norm:.3f}",
                )
            )

        elapsed = time.perf_counter() - epoch_start
        instances_per_sec = (steps_per_epoch * batch_size) / elapsed

        # Test if model beats baseline (lines 11-13)
        test = one_sided_paired_ttest(
            model,
            baseline,
            instance_sampler,
            alpha=alpha,
            device=device,
        )
        if test.improved:
            baseline = prepare_baseline(model)

        summary = EpochSummary(
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
        epoch_summaries.append(summary)

        epoch_bar.set_postfix(
            OrderedDict(
                loss=f"{summary.loss:.4f}",
                advantage=f"{summary.advantage:.3f}",
                obj_model=f"{summary.obj_model:.4f}",
                obj_baseline=f"{summary.obj_baseline:.4f}",
                updated=summary.baseline_updated,
                entropy=f"{summary.entropy:.3f}",
                entropy_norm=f"{summary.entropy_norm:.3f}",
                grad_norm=f"{summary.grad_norm:.3f}",
                log_p=f"{summary.log_p:.3f}",
                ips=f"{summary.instances_per_sec:.1f} i/s",
            )
        )

    return epoch_summaries
