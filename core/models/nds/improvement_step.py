from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from core.constants import EPS
from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.improvement_step import ImprovementStep, _acceptance_to_kernel
from core.lns.operators.repair import RepairOperator
from core.lns.operators.repair.base import NoopRepair
from core.lns.utils import best_repair, select_solution
from core.models.nds.apply_rollout import apply_rollout as default_apply_rollout
from core.models.nds.apply_rollout import only_expanding_apply_rollout
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution

from .models import NDSModel

ApplyRollout = Callable[[BaseSolution, Rollout], BaseSolution]

ImprovementStepType = Literal["sequential", "parallel", "numba_sequential"]


def resolve_improvement_step(
    improvement_step_type: ImprovementStepType,
) -> (
    type[ModelSequentialImprovementStep]
    | type[ModelParallelImprovementStep]
    | type[NumbaModelSequentialImprovementStep]
):
    if improvement_step_type == "sequential":
        return ModelSequentialImprovementStep
    if improvement_step_type == "parallel":
        return ModelParallelImprovementStep
    if improvement_step_type == "numba_sequential":
        return NumbaModelSequentialImprovementStep
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
        first_repair_uses_rollout_order: bool = False,
        insert_in_new_routes_only: bool = False,
    ) -> None:
        if first_repair_uses_rollout_order:
            raise NotImplementedError(
                "first_repair_uses_rollout_order is only supported by "
                "NumbaModelSequentialImprovementStep."
            )
        if insert_in_new_routes_only:
            raise NotImplementedError(
                "insert_in_new_routes_only is only supported by NumbaModelSequentialImprovementStep."
            )
        self._model = model
        self._apply_rollout_fn = (
            only_expanding_apply_rollout
            if isinstance(repair, NoopRepair)
            else default_apply_rollout
        )
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._deconstruction_size = deconstruction_size
        self._n_repairs = n_repairs
        self._stitch = repair.requires_stitch

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
            single_rollout = rollout[..., r]
            deconstructed = self._apply_rollout_fn(current, single_rollout, stitch=self._stitch)
            candidate = best_repair(self._repair, deconstructed, self._n_repairs)
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
        first_repair_uses_rollout_order: bool = False,
        insert_in_new_routes_only: bool = False,
    ) -> None:
        if first_repair_uses_rollout_order:
            raise NotImplementedError(
                "first_repair_uses_rollout_order is only supported by "
                "NumbaModelSequentialImprovementStep."
            )
        if insert_in_new_routes_only:
            raise NotImplementedError(
                "insert_in_new_routes_only is only supported by NumbaModelSequentialImprovementStep."
            )
        self._model = model
        self._apply_rollout_fn = (
            only_expanding_apply_rollout
            if isinstance(repair, NoopRepair)
            else default_apply_rollout
        )
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._deconstruction_size = deconstruction_size
        self._n_repairs = n_repairs
        self._stitch = repair.requires_stitch

    @torch.no_grad()
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        self._model.eval()
        rollout = self._model.rollout(
            solution,
            rollouts_per_solution=self._num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )
        deconstructed = self._apply_rollout_fn(solution, rollout, stitch=self._stitch)
        candidate = best_repair(self._repair, deconstructed, self._n_repairs)

        best_idx = candidate.objective().argmin(dim=-1, keepdim=True)
        best_from_candidate = candidate.gather(-1, best_idx).squeeze(-1)

        mask = accept(solution, best_from_candidate)
        return select_solution(mask, best_from_candidate, solution)


class NumbaModelSequentialImprovementStep(ImprovementStep):
    """CPU/numba sequential step for CVRP driven by model rollouts.

    ``model.rollout`` runs on the GPU (it only uploads features, so a CPU-resident
    solution is fine), producing all rollouts at once. The whole rollout loop
    (stitch-destroy, best-of-n greedy cheapest-insertion and acceptance) then runs
    in a single numba pass over the batch on the CPU, replacing the per-iteration
    GPU kernel launches of ``ModelSequentialImprovementStep``. The repaired
    solution is returned on the same device as the input.

    Only the stitched CVRP greedy-insert repair is supported; the kernel
    reimplements the repair so the ``repair`` argument is validated and otherwise
    unused.
    """

    def __init__(
        self,
        *,
        model: NDSModel,
        repair: RepairOperator,
        num_rollouts: int,
        deconstruction_size: int,
        n_repairs: int,
        seed: int = 0,
        first_repair_uses_rollout_order: bool = False,
        insert_in_new_routes_only: bool = False,
    ) -> None:
        if getattr(repair, "problem_name", None) != "cvrp" or not repair.requires_stitch:
            raise NotImplementedError(
                "NumbaModelSequentialImprovementStep only supports the stitched CVRP "
                "greedy-insert repair."
            )
        self._model = model
        self._num_rollouts = num_rollouts
        self._deconstruction_size = deconstruction_size
        self._n_repairs = n_repairs
        self._seed = seed
        self._first_repair_uses_rollout_order = first_repair_uses_rollout_order
        self._insert_in_new_routes_only = insert_in_new_routes_only

    @torch.no_grad()
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        from core.lns.numba_kernels.cvrp_sequential_improvement_step import run_with_removals
        from core.problems.cvrp.graph_solution import CVRPGraphSolution

        if not isinstance(solution, CVRPGraphSolution):
            raise NotImplementedError(
                f"NumbaModelSequentialImprovementStep not implemented for {type(solution)}"
            )

        self._model.eval()
        rollout = self._model.rollout(
            solution,
            rollouts_per_solution=self._num_rollouts,
            deconstruction_size=self._deconstruction_size,
            greedy=False,
        )

        accept_kind, temperature = _acceptance_to_kernel(accept)
        next, prev, locations, demand, capacity = solution.to_numpy_arrays()
        nodes_to_remove = (
            rollout.actions.reshape(-1, self._num_rollouts, self._deconstruction_size)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )
        next_out, prev_out = run_with_removals(
            next=next,
            prev=prev,
            nodes_to_remove=nodes_to_remove,
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=self._n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=EPS,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_rollout_order,
            insert_in_new_routes_only=self._insert_in_new_routes_only,
        )
        return CVRPGraphSolution.from_numpy_arrays(
            instance=solution.instance,
            next=next_out,
            prev=prev_out,
            batch_size=solution.batch_size,
            device=solution.device,
        )
