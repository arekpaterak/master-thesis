from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class DestroyOperator[S: BaseSolution](ABC):
    @abstractmethod
    def destroy(self, solution: S) -> S: ...

    def __call__(self, solution: S) -> S:
        return self.destroy(solution)
