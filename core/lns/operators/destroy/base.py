from abc import ABC, abstractmethod

import torch

from core.problems.solution import BaseSolution


class DestroyOperator[S: BaseSolution](ABC):
    @abstractmethod
    def __call__(self, solution: S) -> S: ...


class NodeDestroyOperator[S: BaseSolution](DestroyOperator[S], ABC):
    stitch: bool = False

    @abstractmethod
    def select_nodes(self, solution: S, num_rollouts: int) -> torch.Tensor:
        """Return nodes to remove: [*B, R, M]."""
        ...
