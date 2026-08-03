from __future__ import annotations

import torch

from core.problems.cmstp.config import CMSTP
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.sampler import Sampler


class CMSTPSampler(Sampler[CMSTPInstance]):
    problem_name = "cmstp"

    def __init__(self, config: CMSTP, seed: int | None = None) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._initial_state = self._generator.get_state()

    def reset(self) -> None:
        self._generator.set_state(self._initial_state)

    @classmethod
    def for_size(cls, size: int, *, seed: int | None = None) -> CMSTPSampler:
        return cls(CMSTP.for_n(size), seed=seed)

    def __call__(self, batch_size: int) -> CMSTPInstance:
        cfg = self._config
        g = self._generator

        span = cfg.max_coordinate - cfg.min_coordinate
        terminal_locations = (
            torch.rand(batch_size, cfg.num_terminals, 2, generator=g) * span + cfg.min_coordinate
        )
        root_location = torch.rand(batch_size, 1, 2, generator=g) * span + cfg.min_coordinate
        locations = torch.cat([root_location, terminal_locations], dim=1)  # (batch, n+1, 2)

        demand = self._sample_demand(batch_size)  # (batch, n+1), raw
        capacity = torch.full((batch_size, 1), float(cfg.capacity_raw))

        return CMSTPInstance.create(locations=locations, demand=demand, capacity=capacity)

    def _sample_demand(self, batch_size: int) -> torch.Tensor:
        cfg = self._config
        g = self._generator

        demand = torch.randint(
            cfg.min_demand_raw,
            cfg.max_demand_raw + 1,
            (batch_size, cfg.num_terminals),
            generator=g,
        ).float()

        # The root does not have a demand.
        root_demand = torch.zeros(batch_size, 1)
        return torch.cat([root_demand, demand], dim=1)  # (batch, n+1)
