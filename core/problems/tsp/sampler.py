from __future__ import annotations

import torch

from core.problems.sampler import Sampler
from core.problems.tsp import TSP, TSPInstance


class TSPSampler(Sampler[TSPInstance]):
    problem_name = "tsp"

    def __init__(self, config: TSP, seed: int | None = None) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> TSPSampler:
        return cls(TSP(num_locations=size), seed=seed)

    def __call__(self, batch_size: int) -> TSPInstance:
        return TSPInstance.create(
            locations=torch.rand(
                batch_size, self._config.num_locations, 2, generator=self._generator
            )
            * (self._config.max_coordinate - self._config.min_coordinate)
            + self._config.min_coordinate,
        )
