from __future__ import annotations

import torch

from core.problems.sampler import Sampler
from core.problems.tsp import TSPDistribution, TSPInstance


class TSPSampler(Sampler[TSPInstance]):
    problem_name = "tsp"

    def __init__(self, distribution: TSPDistribution, seed: int | None = None) -> None:
        assert distribution.distribution_of_locations == "uniform"

        self._distribution = distribution
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._initial_state = self._generator.get_state()

    def reset(self) -> None:
        self._generator.set_state(self._initial_state)

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> TSPSampler:
        return cls(TSPDistribution(num_locations=size), seed=seed)

    def __call__(self, batch_size: int) -> TSPInstance:
        return TSPInstance.create(
            locations=torch.rand(
                batch_size, self._distribution.num_locations, 2, generator=self._generator
            )
            * (self._distribution.max_coordinate - self._distribution.min_coordinate)
            + self._distribution.min_coordinate,
        )
