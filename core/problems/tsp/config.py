from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TSPConfig:
    num_locations: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    distribution_of_locations: Literal["uniform"] = "uniform"

    def __post_init__(self):
        assert self.num_locations > 0, "Expected num_locations to be a positive integer."
        assert self.min_coordinate < self.max_coordinate, (
            "Expected min_coordinate to be smaller than max_coordinate."
        )
