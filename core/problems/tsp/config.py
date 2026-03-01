from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TSPConfig:
    num_locations: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    distribution_of_locations: Literal["uniform"] = "uniform"
