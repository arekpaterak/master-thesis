from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from core.problems.tsp import TSPConfig


class TSPSampler[T](ABC):
    @abstractmethod
    def __call__(self, batch_size: int) -> T: ...


class TensorTSPSampler(TSPSampler[torch.Tensor]):
    def __init__(self, config: TSPConfig, seed: int | None = None) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, batch_size: int) -> torch.Tensor:
        return (
            torch.rand(batch_size, self._config.num_locations, 2, generator=self._generator)
            * (self._config.max_coordinate - self._config.min_coordinate)
            + self._config.min_coordinate
        )
