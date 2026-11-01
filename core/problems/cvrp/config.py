from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from core.problems.config import BaseProblemConfig

CAPACITY_PER_SIZE: dict[int, int] = {
    10: 20,
    20: 30,
    50: 40,
    100: 50,
    200: 70,
    300: 80,
    500: 100,
    800: 130,
    1000: 200,
    2000: 300,
}


@dataclass(frozen=True)
class CVRPDistribution:
    """Instance distribution for CVRP, consumed by the sampler."""

    num_customers: int
    vehicle_capacity_raw: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    min_demand_raw: int = 1
    max_demand_raw: int = 9
    distribution_of_locations: Literal["uniform"] = "uniform"

    @classmethod
    def for_n(cls, num_customers: int, **kwargs) -> CVRPDistribution:
        """Create a distribution for a given size.

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

    @classmethod
    def without_capacity(cls, num_customers: int, **kwargs) -> CVRPDistribution:
        return cls(
            num_customers=num_customers,
            vehicle_capacity_raw=0,
            min_demand_raw=0,
            max_demand_raw=0,
            **kwargs,
        )


class CVRPConfig(BaseProblemConfig):
    """Input configuration for CVRP."""

    name: Literal["cvrp"] = "cvrp"
    vehicle_capacity_raw: int | None = Field(
        default=None, description="Raw vehicle capacity. If None, derived from size."
    )
    min_demand_raw: int = 1
    max_demand_raw: int = 9
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0

    def to_distribution(self) -> CVRPDistribution:
        common = {
            "min_demand_raw": self.min_demand_raw,
            "max_demand_raw": self.max_demand_raw,
            "min_coordinate": self.min_coordinate,
            "max_coordinate": self.max_coordinate,
        }
        if self.vehicle_capacity_raw is None:
            return CVRPDistribution.for_n(self.size, **common)
        return CVRPDistribution(
            num_customers=self.size,
            vehicle_capacity_raw=self.vehicle_capacity_raw,
            **common,
        )
