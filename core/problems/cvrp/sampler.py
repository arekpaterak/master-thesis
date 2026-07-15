from __future__ import annotations

import torch

from ..sampler import Sampler
from . import CVRPInstance
from .config import CVRP


class CVRPSampler(Sampler[CVRPInstance]):
    problem_name = "cvrp"

    def __init__(self, config: CVRP, seed: int | None = None) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._initial_state = self._generator.get_state()

    def reset(self) -> None:
        self._generator.set_state(self._initial_state)

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> CVRPSampler:
        return cls(CVRP.for_n(size), seed=seed)

    def __call__(self, batch_size: int) -> CVRPInstance:
        cfg = self._config
        g = self._generator

        # Customer coordinates (batch, num_nodes, 2), uniform in a defined square
        customer_locations = (
            torch.rand(batch_size, cfg.num_customers, 2, generator=g)
            * (cfg.max_coordinate - cfg.min_coordinate)
            + cfg.min_coordinate
        )

        # Depot coordinates (batch, 1, 2), uniform in a defined square
        depot_location = (
            torch.rand(batch_size, 1, 2, generator=g) * (cfg.max_coordinate - cfg.min_coordinate)
            + cfg.min_coordinate
        )

        locations = torch.cat([depot_location, customer_locations], dim=1)  # (batch, n+1, 2)

        demand = self._sample_demand(batch_size)  # (batch, n+1), raw

        capacity = torch.full((batch_size, 1), float(cfg.vehicle_capacity_raw))

        return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)

    def _sample_demand(self, batch_size: int) -> torch.Tensor:
        cfg = self._config
        g = self._generator

        # Raw integer demands in [min_demand_raw, max_demand_raw]
        demand = torch.randint(
            cfg.min_demand_raw,
            cfg.max_demand_raw + 1,
            (
                batch_size,
                cfg.num_customers,
            ),
            generator=g,
        ).float()

        # Depot does not have a demand
        depot_demand = torch.zeros(batch_size, 1)

        return torch.cat([depot_demand, demand], dim=1)  # (batch, n+1)
