from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class RepairOperator[S: BaseSolution](ABC):
    @abstractmethod
    def repair(self, partial_solution: S) -> S: ...

    def __call__(self, partial_solution: S) -> S:
        return self.repair(partial_solution)
