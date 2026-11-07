from __future__ import annotations

from abc import abstractmethod

import torch

from core.envs.base import Env
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class ConstructiveEnv[I: BaseProblemInstance, S: BaseSolution, St](Env[I, St]):
    """Build a solution one node at a time; ``to_solution`` decodes the action sequence."""

    @abstractmethod
    def to_solution(self, instance: I, actions: torch.Tensor) -> S: ...

    def reward(self, instance: I, actions: torch.Tensor) -> torch.Tensor:
        return -self.to_solution(instance, actions).objective()
