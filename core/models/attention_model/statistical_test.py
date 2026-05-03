from typing import NamedTuple

import torch
from scipy import stats

from core.models.attention_model import AttentionModel
from core.models.attention_model.evaluate import evaluate
from core.problems.sampler import Sampler


class OneSidedPairedTTestResult(NamedTuple):
    improved: bool
    obj_model_mean: float
    obj_baseline_mean: float


def one_sided_paired_ttest(
    model: AttentionModel,
    baseline: AttentionModel,
    instance_sampler: Sampler,
    n_instances: int = 10_000,
    batch_size: int = 512,
    alpha: float = 0.05,
    device: str | torch.types.Device = "cpu",
) -> OneSidedPairedTTestResult:
    """One-sided paired t-test: is model significantly better than baseline?"""
    obj_model = evaluate(
        model, instance_sampler, n_instances=n_instances, batch_size=batch_size, device=device
    ).objective
    obj_baseline = evaluate(
        baseline, instance_sampler, n_instances=n_instances, batch_size=batch_size, device=device
    ).objective
    t, p = stats.ttest_rel(obj_model.numpy(), obj_baseline.numpy())
    p_one_sided = p / 2 if t < 0 else 1.0
    return OneSidedPairedTTestResult(
        improved=(p_one_sided < alpha),
        obj_model_mean=obj_model.mean().item(),
        obj_baseline_mean=obj_baseline.mean().item(),
    )
