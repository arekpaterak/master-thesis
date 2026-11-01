from __future__ import annotations

from typing import Protocol

import torch

from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.instance import TSPInstance
from core.problems.tsp.solution import TSPSolution

__all__ = [
    "STRATEGIES",
    "TSPSolveStrategy",
    "random_permutation",
    "random_permutation_graph",
]


class TSPSolveStrategy(Protocol):
    def __call__(self, instance: TSPInstance) -> TSPSolution: ...


def random_permutation(instance: TSPInstance, *, seed: int | None = None) -> TSPSolution:
    was_batched = True
    if len(instance.batch_size) == 0:
        instance = instance.unsqueeze(0)
        was_batched = False
    batch_size = instance.batch_size[0]
    num_locations = instance.num_locations
    device = instance.locations.device

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    random_permutations = torch.argsort(
        torch.rand(batch_size, num_locations, generator=generator).to(device=device), dim=-1
    )

    if not was_batched:
        instance = instance.squeeze(0)
        random_permutations = random_permutations.squeeze(0)

    return TSPSolution.create(
        instance=instance,
        tour=random_permutations,
    )


def random_permutation_graph(instance: TSPInstance, *, seed: int | None = None) -> TSPGraphSolution:
    solution = random_permutation(instance, seed=seed)
    return TSPGraphSolution.create_from_sequence(instance=instance, sequence=solution.tour)


STRATEGIES = [
    random_permutation,
]
