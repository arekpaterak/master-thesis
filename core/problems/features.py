from typing import Protocol, TypeVar

import torch

from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution

SolutionType = TypeVar("SolutionType", bound=BaseSolution)
InstanceType = TypeVar("InstanceType", bound=BaseProblemInstance)


class FeatureExtractor(Protocol[SolutionType]):
    """Protocol for extracting neural network input features from a solution."""

    def __call__(self, solution: SolutionType) -> dict[str, torch.Tensor]: ...


class InstanceFeatureExtractor(Protocol[InstanceType]):
    """Protocol for extracting neural network input features from an instance."""

    def __call__(self, instance: InstanceType) -> dict[str, torch.Tensor]: ...
