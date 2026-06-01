from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.improvement_step import ImprovementStep
from core.lns.operators.repair import RepairOperator
from core.lns.utils import select_solution
from core.models.nds.functions import apply_rollout as default_apply_rollout
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution

from .models import NDSModel

ApplyRollout = Callable[[BaseSolution, Rollout], BaseSolution]

ImprovementStepType = Literal["sequential", "parallel"]


def resolve_improvement_step(
    improvement_step_type: ImprovementStepType,
) -> type[ModelSequentialImprovementStep] | type[ModelParallelImprovementStep]:
    if improvement_step_type == "sequential":
        return ModelSequentialImprovementStep
    if improvement_step_type == "parallel":
        return ModelParallelImprovementStep
    raise ValueError(f"Unknown improvement_step_type: {improvement_step_type!r}")


class ModelSequentialImprovementStep(ImprovementStep):
    def __init__(
        self,
        *,
        model: NDSModel,
        repair: RepairOperator,
        num_rollouts: int,
        deconstruction_size: int,
        n_repairs: int,
    ) -> None:
        self._model = model
        self._apply_rollout_fn = default_apply_rollout
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._deconstruction_size = deconstruction_size
        self._n_repairs = n_repairs

    @torch.no_grad()
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        self._model.eval()
        rollout = self._model.rollout(
            solution,
            rollouts_per_solution=self._num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )
        current = solution
        for r in range(self._num_rollouts):
            single_rollout = Rollout.create(
                actions=rollout.actions[:, r, :],
                log_prob=rollout.log_p[:, r],
                entropy=rollout.entropy[:, r],
                entropy_norm=rollout.entropy_norm[:, r],
            )
            deconstructed = self._apply_rollout_fn(current, single_rollout)
            candidate = self._repair(deconstructed)
            for _ in range(self._n_repairs - 1):
                repaired = self._repair(deconstructed)
                candidate = select_solution(
                    repaired.objective() < candidate.objective(), repaired, candidate
                )
            mask = accept(current, candidate)
            current = select_solution(mask, candidate, current)
        return current


class ModelParallelImprovementStep(ImprovementStep):
    def __init__(
        self,
        *,
        model: NDSModel,
        repair: RepairOperator,
        num_rollouts: int,
        deconstruction_size: int,
        n_repairs: int,
    ) -> None:
        self._model = model
        self._apply_rollout_fn = default_apply_rollout
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._deconstruction_size = deconstruction_size
        self._n_repairs = n_repairs

    @torch.no_grad()
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        self._model.eval()
        rollout = self._model.rollout(
            solution,
            rollouts_per_solution=self._num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )
        deconstructed = self._apply_rollout_fn(solution, rollout)
        candidate = self._repair(deconstructed)
        for _ in range(self._n_repairs - 1):
            repaired = self._repair(deconstructed)
            candidate = select_solution(
                repaired.objective() < candidate.objective(), repaired, candidate
            )

        best_idx = candidate.objective().argmin(dim=-1, keepdim=True)
        best_from_candidate = candidate.gather(-1, best_idx).squeeze(-1)

        mask = accept(solution, best_from_candidate)
        return select_solution(mask, best_from_candidate, solution)
