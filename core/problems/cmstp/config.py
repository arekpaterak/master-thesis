from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from core.problems.config import BaseProblemConfig
from core.problems.cvrp.config import CAPACITY_PER_SIZE


@dataclass(frozen=True)
class CMSTPDistribution:
    """Instance distribution for CMSTP, consumed by the sampler.

    Fields mirror ``CVRPDistribution``: the instance data (depot + customers + demand + capacity) is
    identical; only the solution built over it differs (a capacitated spanning tree rooted
    at node 0 instead of a set of routes).
    """

    num_customers: int
    capacity_raw: int
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0
    min_demand_raw: int = 1
    max_demand_raw: int = 9
    distribution_of_locations: Literal["uniform"] = "uniform"

    @classmethod
    def for_n(cls, num_customers: int, **kwargs) -> CMSTPDistribution:
        """Create a distribution for a given size, reusing the CVRP capacity schedule."""
        if num_customers not in CAPACITY_PER_SIZE:
            raise ValueError(
                f"No standard capacity for n={num_customers}. "
                f"Supported sizes: {sorted(CAPACITY_PER_SIZE)}. "
                "Pass capacity_raw explicitly to use a custom size."
            )
        return cls(
            num_customers=num_customers,
            capacity_raw=CAPACITY_PER_SIZE[num_customers],
            **kwargs,
        )


class CMSTPConfig(BaseProblemConfig):
    """Input configuration for CMSTP."""

    name: Literal["cmstp"] = "cmstp"
    capacity_raw: int | None = Field(
        default=None, description="Raw per-subtree capacity. If None, derived from size."
    )
    min_demand_raw: int = 1
    max_demand_raw: int = 9
    min_coordinate: float = 0.0
    max_coordinate: float = 1.0

    def to_distribution(self) -> CMSTPDistribution:
        common = {
            "min_demand_raw": self.min_demand_raw,
            "max_demand_raw": self.max_demand_raw,
            "min_coordinate": self.min_coordinate,
            "max_coordinate": self.max_coordinate,
        }
        if self.capacity_raw is None:
            return CMSTPDistribution.for_n(self.size, **common)
        return CMSTPDistribution(
            num_customers=self.size,
            capacity_raw=self.capacity_raw,
            **common,
        )
