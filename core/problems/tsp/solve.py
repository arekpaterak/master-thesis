from typing import Protocol

import torch

from core.problems.tsp import TSPBatchInstance
from core.problems.tsp.solution import TSPBatchSolution


class TSPSolveStrategy(Protocol):
    def __call__(self, instance: TSPBatchInstance) -> TSPBatchSolution: ...


def random_permutation(instance: TSPBatchInstance) -> TSPBatchSolution:
    batch_size, num_locations, _ = instance.locations.shape

    customer_indices = torch.arange(0, num_locations, device=instance.locations.device)
    random_permutations = torch.stack(
        [customer_indices[torch.randperm(num_locations)] for _ in range(batch_size)]
    )

    return TSPBatchSolution(locations=instance.locations, tour=random_permutations)
