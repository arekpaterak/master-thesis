from abc import ABC, abstractmethod

import torch

from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


class SolutionFeatureExtractor[S: BaseSolution](ABC):
    """Abstract base class for extracting neural network input features from a solution."""

    @abstractmethod
    def __call__(self, solution: S) -> dict[str, torch.Tensor]: ...


class InstanceFeatureExtractor[I: BaseProblemInstance](ABC):
    """Abstract base class for extracting neural network input features from an instance."""

    @abstractmethod
    def __call__(self, instance: I) -> dict[str, torch.Tensor]: ...
