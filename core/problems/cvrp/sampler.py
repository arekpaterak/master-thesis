from __future__ import annotations

import torch

from ..sampler import Sampler
from . import CVRPInstance
from .config import CVRPConfig


class CVRPSampler(Sampler[CVRPInstance]):
    def __init__(self, config: CVRPConfig, seed: int | None = None) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, batch_size: int) -> CVRPInstance:
        cfg = self._config
        g = self._generator

        # Customer coordinates (batch, num_nodes, 2), uniform in a unit square
        customer_locations = (
            torch.rand(batch_size, cfg.num_customers, 2, generator=g)
            * (cfg.max_coordinate - cfg.min_coordinate)
            + cfg.min_coordinate
        )

        # Depot coordinates (batch, 1, 2), uniform in a unit square
        depot_location = (
            torch.rand(batch_size, 1, 2, generator=g) * (cfg.max_coordinate - cfg.min_coordinate)
            + cfg.min_coordinate
        )

        locations = torch.cat([depot_location, customer_locations], dim=1)  # (batch, n+1, 2)

        demand = self._sample_demand(batch_size)  # (batch, n+1)

        # After normalisation of the demands, the effective capacity is 1.0
        capacity = torch.ones(batch_size, 1)

        return CVRPInstance.create(locations=locations, demand=demand, capacity=capacity)

    def _sample_demand(self, batch_size: int) -> torch.Tensor:
        cfg = self._config
        g = self._generator

        demand = torch.randint(
            cfg.min_demand_raw,
            cfg.max_demand_raw + 1,
            (
                batch_size,
                cfg.num_customers,
            ),
            generator=g,
        ).float()

        # Normalised demands from (0.0, 1.0]
        norm_demand = demand / cfg.vehicle_capacity_raw

        # Depot does not have a demand
        depot_demand = torch.zeros(batch_size, 1)

        return torch.cat([depot_demand, norm_demand], dim=1)  # (batch, n+1)
