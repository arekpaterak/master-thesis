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
    num_customers: int
    vehicle_capacity_raw: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    min_demand_raw: int = 1
    max_demand_raw: int = 9
    distribution_of_locations: Literal["uniform"] = "uniform"

    @classmethod
    def for_n(cls, num_customers: int, **kwargs) -> CVRPConfig:
        """Create a config for a given problem size.

        Raises ``ValueError`` for unsupported sizes.
        """
        if num_customers not in CAPACITY_PER_SIZE:
            raise ValueError(
                f"No standard capacity for n={num_customers}. "
                f"Supported sizes: {sorted(CAPACITY_PER_SIZE)}. "
                "Pass vehicle_capacity_raw explicitly to use a custom size."
            )
        return cls(
            num_customers=num_customers,
            vehicle_capacity_raw=CAPACITY_PER_SIZE[num_customers],
            **kwargs,
        )
