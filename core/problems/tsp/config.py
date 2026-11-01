from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.problems.config import BaseProblemConfig


@dataclass(frozen=True)
class TSPDistribution:
    """Instance distribution for TSP, consumed by the sampler."""

    num_locations: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    distribution_of_locations: Literal["uniform"] = "uniform"

    def __post_init__(self):
        assert self.num_locations > 0, "Expected num_locations to be a positive integer."
        assert self.min_coordinate < self.max_coordinate, (
            "Expected min_coordinate to be smaller than max_coordinate."
        )


class TSPConfig(BaseProblemConfig):
    """Input configuration for TSP."""

    name: Literal["tsp"] = "tsp"
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0

    def to_distribution(self) -> TSPDistribution:
        return TSPDistribution(
            num_locations=self.size,
            min_coordinate=self.min_coordinate,
            max_coordinate=self.max_coordinate,
        )
