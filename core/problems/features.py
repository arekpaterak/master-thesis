from typing import Protocol, TypeVar

import torch

from core.problems.solution import BaseSolution

SolutionType = TypeVar("SolutionType", bound=BaseSolution)


class FeatureExtractor(Protocol[SolutionType]):
    """Protocol for extracting neural network input features from a solution."""

    def __call__(self, solution: SolutionType) -> dict[str, torch.Tensor]: ...
