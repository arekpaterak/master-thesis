from abc import ABC, abstractmethod
from typing import ClassVar

from tensordict.tensorclass import TensorClass


class BaseProblemInstance(TensorClass, ABC):
    num_fixed_nodes: ClassVar[int]
    """Leading nodes that can never be removed from a solution (e.g. the depot)."""

    @property
    @abstractmethod
    def num_nodes(self) -> int: ...

    @property
    @abstractmethod
    def problem_size(self) -> int: ...

    @abstractmethod
    def normalize_(self) -> None: ...

    @abstractmethod
    def visualize(self, **kwargs) -> None: ...
