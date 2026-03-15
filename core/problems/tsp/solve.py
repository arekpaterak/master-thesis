from __future__ import annotations

from typing import Protocol

import torch

from core.problems.tsp import TSPBatchInstance
from core.problems.tsp.solution import TSPBatchSolution


class TSPSolveStrategy(Protocol):
    def __call__(self, instance: TSPBatchInstance) -> TSPBatchSolution: ...


def random_permutation(instance: TSPBatchInstance) -> TSPBatchSolution:
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False
    batch_size = instance.batch_size[0]
    num_locations = instance.num_locations

    device = instance.locations.device

    random_permutations = torch.argsort(
        torch.rand(batch_size, num_locations, device=device), dim=-1
    )

    if not was_batched:
        instance = instance.squeeze(0)
        random_permutations = random_permutations.squeeze(0)

    return TSPBatchSolution(
        instance=instance,
        tour=random_permutations,
        batch_size=instance.batch_size,
    )


STRATEGIES = [
    random_permutation,
]
