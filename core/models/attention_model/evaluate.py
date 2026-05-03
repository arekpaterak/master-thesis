from typing import NamedTuple

import torch

from core.models.attention_model import AttentionModel
from core.models.attention_model.model import rollout_to_solution
from core.problems.sampler import Sampler


class EvaluationResult(NamedTuple):
    objective: torch.Tensor


def evaluate(
    model: AttentionModel,
    instance_sampler: Sampler,
    *,
    n_instances: int = 10_000,
    batch_size: int = 512,
    device: str | torch.types.Device = "cpu",
) -> EvaluationResult:
    model.eval()
    objectives = []
    with torch.no_grad():
        for _ in range(0, n_instances, batch_size):
            effective_batch_size = min(batch_size, n_instances - len(objectives) * batch_size)
            instance = instance_sampler(effective_batch_size).to(device)
            rollout = model.rollout(instance, greedy=True)
            solution = rollout_to_solution(rollout, instance)
            objectives.append(solution.objective().cpu())
    return EvaluationResult(objective=torch.cat(objectives))
