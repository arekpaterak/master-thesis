from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.constants import EPS
from core.problems.solution import BaseSolution

from .acceptance_criterion import (
    AcceptanceCriterion,
    GreedyAcceptance,
    SimulatedAnnealingAcceptance,
)
from .operators.destroy.base import DestroyOperator
from .operators.repair.base import RepairOperator
from .utils import best_repair, select_solution

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


class NumbaSequentialImprovementStep(ImprovementStep):
    """CPU/numba sequential step for CVRP with in-kernel random-node destroy.

    Folds the whole rollout loop (random-node destroy, best-of-n greedy
    cheapest-insertion and acceptance) into a single numba pass over the batch,
    avoiding the per-iteration GPU kernel launches of ``SequentialImprovementStep``.
    Only supports a stitched CVRP greedy-insert repair (the kernel reimplements the
    repair, so the ``repair`` argument is validated and otherwise unused) paired
    with ``RandomNodeDestroy``.
    """

    def __init__(
        self,
        *,
        destroy: DestroyOperator,
        repair: RepairOperator,
        num_rollouts: int = 1,
        n_repairs: int = 1,
        seed: int = 0,
        insert_in_new_routes_only: bool = False,
    ) -> None:
        from .operators.destroy.random_node import RandomNodeDestroy

        if not isinstance(destroy, RandomNodeDestroy):
            raise NotImplementedError(
                "NumbaSequentialImprovementStep only supports RandomNodeDestroy."
            )
        if getattr(repair, "problem_name", None) != "cvrp" or not repair.requires_stitch:
            raise NotImplementedError(
                "NumbaSequentialImprovementStep only supports the stitched CVRP greedy-insert repair."
            )
        self._destruction_size = destroy.destruction_size
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs
        self._seed = seed
        self._insert_in_new_routes_only = insert_in_new_routes_only

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        from core.lns.numba_kernels.cvrp_sequential_improvement_step import run_with_random
        from core.problems.cvrp.graph_solution import CVRPGraphSolution

        if not isinstance(solution, CVRPGraphSolution):
            raise NotImplementedError(
                f"NumbaSequentialImprovementStep not implemented for {type(solution)}"
            )

        accept_kind, temperature = _acceptance_to_kernel(accept)
        next_np, prev_np, locations, demand, capacity = solution.to_numpy_arrays()
        next_out, prev_out = run_with_random(
            next=next_np,
            prev=prev_np,
            locations=locations,
            demand=demand,
            capacity=capacity,
            num_rollouts=self._num_rollouts,
            destruction_size=self._destruction_size,
            n_repairs=self._n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=EPS,
            base_seed=self._seed,
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
    """Build a random-node-destroy improvement step from a search config.

    All knobs (``repair_type``, ``improvement_step_type``, ``deconstruction_size``,
    ``num_rollouts``, ``n_repairs``) are read from ``search``; callers that need different
    knobs (e.g. the start-solution warmup) pass a modified copy of the search config.
    """
    # Lazy import: top-level would create a cycle (random_node -> problems -> ... -> lns).
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
    """Map an acceptance criterion to the (kind, temperature) the numba kernel expects.

    ``kind`` is 0 for greedy (strict improvement) and 1 for simulated annealing.
    """
    if isinstance(accept, GreedyAcceptance):
        return 0, 0.0
    if isinstance(accept, SimulatedAnnealingAcceptance):
        return 1, float(accept.temperature)
    raise NotImplementedError(
        f"Numba ImprovementSteps support GreedyAcceptance and SimulatedAnnealingAcceptance, got {type(accept)}"
    )
