from __future__ import annotations

from abc import ABC, abstractmethod

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.problems.solution import BaseSolution


class ImprovementStep(ABC):
    @abstractmethod
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution: ...


class NoopImprovementStep(ImprovementStep):
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        return solution
