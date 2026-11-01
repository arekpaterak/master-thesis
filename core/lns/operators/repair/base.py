from __future__ import annotations

from abc import ABC, abstractmethod

from core.problems.solution import BaseSolution


class RepairOperator[S: BaseSolution](ABC):
    requires_stitch: bool = False

    @abstractmethod
    def __call__(self, deconstructed_solution: S) -> S: ...
