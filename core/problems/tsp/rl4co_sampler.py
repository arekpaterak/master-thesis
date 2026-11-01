from __future__ import annotations

from rl4co.envs.routing.tsp.generator import TSPGenerator
from tensordict import TensorDict
from torch.distributions import Uniform

from core.problems.sampler import Sampler
from core.problems.tsp.config import TSPDistribution


class RL4COTSPSampler(Sampler[TensorDict]):
    def __init__(self, distribution: TSPDistribution) -> None:
        assert distribution.distribution_of_locations == "uniform"

        self._distribution = distribution
        self._generator = TSPGenerator(
            num_loc=distribution.num_locations,
            min_loc=distribution.min_coordinate,
            max_loc=distribution.max_coordinate,
            loc_distribution=Uniform,
        )

    def __call__(self, batch_size: int) -> TensorDict:
        return self._generator([batch_size])

    @property
    def generator(self) -> TSPGenerator:
        return self._generator
