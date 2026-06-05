from __future__ import annotations

from abc import ABC, abstractmethod

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.operators.destroy.base import DestroyOperator
from core.lns.operators.repair.base import RepairOperator
from core.lns.utils import best_repair, select_solution
from core.problems.solution import BaseSolution


class ImprovementStep(ABC):
    @abstractmethod
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution: ...


class NoopImprovementStep(ImprovementStep):
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        return solution


class SequentialImprovementStep(ImprovementStep):
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
            deconstructed = self._destroy(current)
            candidate = best_repair(self._repair, deconstructed, self._n_repairs)
            mask = accept(current, candidate)
            current = select_solution(mask, candidate, current)
        return current


class ParallelImprovementStep(ImprovementStep):
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
        R = self._num_rollouts
        expanded = solution.unsqueeze(-1).expand(*solution.batch_size, R)

        deconstructed = self._destroy(expanded)
        candidate = best_repair(self._repair, deconstructed, self._n_repairs)
        best_idx = candidate.objective().argmin(dim=-1, keepdim=True)
        best_from_candidate = candidate.gather(-1, best_idx).squeeze(-1)
        mask = accept(solution, best_from_candidate)
        return select_solution(mask, best_from_candidate, solution)
