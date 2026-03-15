from __future__ import annotations

from rl4co.envs.routing.tsp.generator import TSPGenerator
from tensordict import TensorDict
from torch.distributions import Uniform

from core.problems.sampler import Sampler
from core.problems.tsp.config import TSPConfig


class RL4COTSPSampler(Sampler[TensorDict]):
    def __init__(self, config: TSPConfig) -> None:
        assert config.distribution_of_locations == "uniform"

        self._config = config
        self._generator = TSPGenerator(
            num_loc=config.num_locations,
            min_loc=config.min_coordinate,
            max_loc=config.max_coordinate,
            loc_distribution=Uniform,
        )

    def __call__(self, batch_size: int) -> TensorDict:
        return self._generator([batch_size])

    @property
    def generator(self) -> TSPGenerator:
        return self._generator
