from __future__ import annotations

from abc import ABC, abstractmethod

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.operators.destroy.base import DestroyOperator
from core.lns.operators.repair.base import RepairOperator
from core.lns.utils import select_solution
from core.problems.solution import BaseSolution


class ImprovementStep(ABC):
    @abstractmethod
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution: ...


class NoopImprovementStep(ImprovementStep):
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        return solution


class DestroyRepairImprovementStep(ImprovementStep):
    def __init__(
        self,
        *,
        destroy: DestroyOperator,
        repair: RepairOperator,
        num_rollouts: int = 1,
        n_repairs: int = 1,
    ) -> None:
        self._destroy = destroy
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        current = solution
        for _ in range(self._num_rollouts):
            deconstructed = self._destroy(solution)
            candidate = self._repair(deconstructed)
            for _ in range(self._n_repairs - 1):
                repaired = self._repair(deconstructed)
                candidate = select_solution(
                    repaired.objective() < candidate.objective(), repaired, candidate
                )
            mask = accept(current, candidate)
            current = select_solution(mask, candidate, current)
        return current
