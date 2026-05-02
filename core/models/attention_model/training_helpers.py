import copy
from collections.abc import Callable
from typing import NamedTuple

import torch
from scipy import stats

from core.models.attention_model import AttentionModel
from core.models.utils import freeze_parameters_
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


def sync_baseline_(baseline: AttentionModel, model: AttentionModel) -> None:
    baseline.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline.eval()
    freeze_parameters_(baseline)


def prepare_baseline(model: AttentionModel) -> AttentionModel:
    baseline = copy.deepcopy(model)
    sync_baseline_(baseline, model)
    return baseline
