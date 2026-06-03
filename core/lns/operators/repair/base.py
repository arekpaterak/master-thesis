from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class RepairOperator[S: BaseSolution](ABC):
    requires_stitch: bool = False

    @abstractmethod
    def __call__(self, deconstructed_solution: S) -> S: ...


# TODO: Remove as this might not produce complete solution
class NoopRepair[S: BaseSolution](RepairOperator[S]):
    def __call__(self, deconstructed_solution: S) -> S:
        return deconstructed_solution
