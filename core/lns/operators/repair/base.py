from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class RepairOperator[S: BaseSolution](ABC):
    @abstractmethod
    def __call__(self, partial_solution: S) -> S: ...


# TODO: Remove as this might not produce complete solution
class NoopRepair[S: BaseSolution](RepairOperator[S]):
    def __call__(self, partial_solution: S) -> S:
        return partial_solution
