from abc import ABC, abstractmethod

from tensordict.tensorclass import TensorClass


class BaseProblemInstance(TensorClass, ABC):
    ...

    @property
    @abstractmethod
    def num_nodes(self) -> int: ...

    @abstractmethod
    def visualize(self, **kwargs) -> None: ...
