from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CAPACITY_PER_SIZE: dict[int, int] = {
    20: 30,
    50: 40,
    100: 50,
}


@dataclass(frozen=True)
class CVRPConfig:
    num_locations: int
    vehicle_capacity: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    min_demand: int = 1
    max_demand: int = 9
    distribution_of_locations: Literal["uniform"] = "uniform"

    @classmethod
    def for_n(cls, num_nodes: int, **kwargs) -> CVRPConfig:
        """Create a config for a given problem size.

        Raises ``ValueError`` for unsupported sizes.
        """
        if num_nodes not in CAPACITY_PER_SIZE:
            raise ValueError(
                f"No standard capacity for n={num_nodes}. "
                f"Supported sizes: {sorted(CAPACITY_PER_SIZE)}. "
                "Pass vehicle_capacity explicitly to use a custom size."
            )
        return cls(
            num_locations=num_nodes,
            vehicle_capacity=CAPACITY_PER_SIZE[num_nodes],
            **kwargs,
        )
