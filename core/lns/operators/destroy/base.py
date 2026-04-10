from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class DestroyOperator[S: BaseSolution](ABC):
    @abstractmethod
    def __call__(self, solution: S) -> S: ...
