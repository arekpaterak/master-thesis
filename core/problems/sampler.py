from abc import ABC, abstractmethod

from core.problems.instance import BaseProblemInstance


class Sampler[T: BaseProblemInstance](ABC):
    @abstractmethod
    def __call__(self, batch_size: int) -> T: ...
