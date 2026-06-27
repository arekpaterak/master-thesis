from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from core.constants import EPS
from core.problems.solution import BaseSolution

from .acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptance,
    SimulatedAnnealingAcceptance,
)
from .operators.destroy.base import NodeDestroyOperator
from .operators.repair.base import RepairOperator
from .utils import apply_nodes, best_repair, select_solution

if TYPE_CHECKING:
    from .config import SearchParams


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
        destroy: NodeDestroyOperator,
        repair: RepairOperator,
        num_rollouts: int = 1,
        n_repairs: int = 1,
    ) -> None:
        self._destroy = destroy
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        current = solution
        for r in range(self._num_rollouts):
            deconstructed = apply_nodes(current, nodes[..., r, :], stitch=self._destroy.stitch)
            candidate = best_repair(self._repair, deconstructed, self._n_repairs)
            mask = accept(current, candidate)
            current = select_solution(mask, candidate, current)
        return current


class ParallelImprovementStep(ImprovementStep):
    def __init__(
        self,
        *,
        destroy: NodeDestroyOperator,
        repair: RepairOperator,
        num_rollouts: int = 1,
        n_repairs: int = 1,
    ) -> None:
        self._destroy = destroy
        self._repair = repair
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        deconstructed = apply_nodes(solution, nodes, stitch=self._destroy.stitch)
        candidate = best_repair(self._repair, deconstructed, self._n_repairs)
        best_idx = candidate.objective().argmin(dim=-1, keepdim=True)
        best_from_candidate = candidate.gather(-1, best_idx).squeeze(-1)
        mask = accept(solution, best_from_candidate)
        return select_solution(mask, best_from_candidate, solution)


class NumbaSequentialImprovementStep(ImprovementStep):
    """CPU/numba sequential step for CVRP.

    Accepts any :class:`NodeDestroyOperator`. Calls ``select_nodes`` to obtain
    all rollout node sets upfront, then folds the whole rollout loop (stitch-destroy,
    best-of-n greedy cheapest-insertion and acceptance) into a single numba pass,
    avoiding per-iteration GPU kernel launches.

    Only the stitched CVRP greedy-insert repair is supported (the kernel reimplements
    the repair, so the ``repair`` argument is validated and otherwise unused).
    """

    def __init__(
        self,
        *,
        destroy: NodeDestroyOperator,
        repair: RepairOperator,
        num_rollouts: int = 1,
        n_repairs: int = 1,
        seed: int = 0,
        first_repair_uses_rollout_order: bool = False,
        insert_in_new_routes_only: bool = False,
    ) -> None:
        if getattr(repair, "problem_name", None) != "cvrp" or not repair.requires_stitch:
            raise NotImplementedError(
                "NumbaSequentialImprovementStep only supports the stitched CVRP greedy-insert repair."
            )
        self._destroy = destroy
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs
        self._seed = seed
        self._first_repair_uses_rollout_order = first_repair_uses_rollout_order
        self._insert_in_new_routes_only = insert_in_new_routes_only

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        from core.lns.numba_kernels.cvrp_sequential_improvement_step import run_with_removals
        from core.problems.cvrp.graph_solution import CVRPGraphSolution

        if not isinstance(solution, CVRPGraphSolution):
            raise NotImplementedError(
                f"NumbaSequentialImprovementStep not implemented for {type(solution)}"
            )

        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        accept_kind, temperature = _acceptance_to_kernel(accept)
        next_np, prev_np, locations, demand, capacity = solution.to_numpy_arrays()
        nodes_to_remove = (
            nodes.reshape(-1, nodes.shape[-2], nodes.shape[-1])
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )
        next_out, prev_out = run_with_removals(
            next=next_np,
            prev=prev_np,
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


def build_random_destroy_step(
    search: SearchParams, *, problem_name: str, seed: int | None
) -> ImprovementStep:
    """Build a random-node-destroy improvement step from a search config."""
    from .operators.destroy.random_node import RandomNodeDestroy

    repair = RepairOperator.create(
        problem_name, search.repair_type, deconstruction_size=search.deconstruction_size
    )
    destroy = RandomNodeDestroy(
        destruction_size=search.deconstruction_size,
        seed=seed,
        stitch=repair.requires_stitch,
    )

    match search.improvement_step_type:
        case "numba_sequential":
            return NumbaSequentialImprovementStep(
                destroy=destroy,
                repair=repair,
                num_rollouts=search.num_rollouts,
                n_repairs=search.n_repairs,
                seed=seed or 0,
            )
        case "sequential":
            return SequentialImprovementStep(
                destroy=destroy,
                repair=repair,
                num_rollouts=search.num_rollouts,
                n_repairs=search.n_repairs,
            )
        case "parallel":
            return ParallelImprovementStep(
                destroy=destroy,
                repair=repair,
                num_rollouts=search.num_rollouts,
                n_repairs=search.n_repairs,
            )
        case _:
            raise NotImplementedError("Unknown improvement step type.")


def _acceptance_to_kernel(accept: AcceptanceCriterion) -> tuple[int, float]:
    """Map an acceptance criterion to the (kind, temperature) the numba kernel expects."""
    if isinstance(accept, GreedyAcceptance):
        return 0, 0.0
    if isinstance(accept, SimulatedAnnealingAcceptance):
        return 1, float(accept.temperature)
    raise NotImplementedError(
        f"Numba ImprovementSteps support GreedyAcceptance and SimulatedAnnealingAcceptance, got {type(accept)}"
    )
