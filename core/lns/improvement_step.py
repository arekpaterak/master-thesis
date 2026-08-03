from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
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
from .utils import best_repair, remove_nodes, select_solution

if TYPE_CHECKING:
    from core.problems.cmstp.solution import CMSTPSolution
    from core.problems.cvrp.graph_solution import CVRPGraphSolution
    from core.problems.tsp.graph_solution import TSPGraphSolution

    from .config import SearchParams


class ImprovementStep(ABC):
    @abstractmethod
    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution: ...

    @property
    def _operators_with_search_sessions(self) -> tuple[NodeDestroyOperator, ...]:
        """Operators whose per-search session should be opened around a search."""
        return ()

    @contextmanager
    def search_session(self, solution: BaseSolution) -> Iterator[None]:
        with ExitStack() as stack:
            for operator in self._operators_with_search_sessions:
                stack.enter_context(operator.search_session(solution))
            yield


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

    @property
    def _operators_with_search_sessions(self) -> tuple[NodeDestroyOperator, ...]:
        return (self._destroy,)

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        current = solution
        for r in range(self._num_rollouts):
            deconstructed = remove_nodes(current, nodes[..., r, :], stitch=self._destroy.stitch)
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

    @property
    def _operators_with_search_sessions(self) -> tuple[NodeDestroyOperator, ...]:
        return (self._destroy,)

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        deconstructed = remove_nodes(solution, nodes, stitch=self._destroy.stitch)
        candidate = best_repair(self._repair, deconstructed, self._n_repairs)
        best_idx = candidate.objective().argmin(dim=-1, keepdim=True)
        best_from_candidate = candidate.gather(-1, best_idx).squeeze(-1)
        mask = accept(solution, best_from_candidate)
        return select_solution(mask, best_from_candidate, solution)


class NumbaSequentialImprovementStep(ImprovementStep):
    """CPU/numba sequential step.

    Accepts any :class:`NodeDestroyOperator`. Calls ``select_nodes`` to obtain
    all rollout node sets upfront, then folds the whole rollout loop (stitch-destroy,
    best-of-n greedy cheapest-insertion and acceptance) into a single numba pass,
    avoiding per-iteration GPU kernel launches.

    Only the stitched CVRP greedy-insert repair or the TSP greedy-insert repair are supported
    (the kernel reimplements the repair, so the ``repair`` argument is validated and otherwise unused).
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
        self._validate_repair(repair)
        self._destroy = destroy
        self._num_rollouts = num_rollouts
        self._n_repairs = n_repairs
        self._seed = seed
        self._first_repair_uses_rollout_order = first_repair_uses_rollout_order
        self._insert_in_new_routes_only = insert_in_new_routes_only

    @property
    def _operators_with_search_sessions(self) -> tuple[NodeDestroyOperator, ...]:
        return (self._destroy,)

    @staticmethod
    def _validate_repair(repair: RepairOperator) -> None:
        match getattr(repair, "problem_name", None):
            case "cvrp" if repair.repair_type == "greedy_insert" and repair.requires_stitch:
                return
            case "tsp" if repair.repair_type == "greedy_insert":
                return
            case "cmstp" if repair.repair_type == "greedy_insert" and repair.requires_stitch:
                return
        raise NotImplementedError(
            "NumbaSequentialImprovementStep only supports the stitched CVRP "
            "greedy-insert repair, the TSP greedy-insert repair, or the stitched CMSTP "
            f"greedy-insert repair, got {type(repair).__name__} "
            f"(problem_name={getattr(repair, 'problem_name', None)!r})."
        )

    def __call__(self, solution: BaseSolution, accept: AcceptanceCriterion) -> BaseSolution:
        from core.problems.cmstp.solution import CMSTPSolution
        from core.problems.cvrp.graph_solution import CVRPGraphSolution
        from core.problems.tsp.graph_solution import TSPGraphSolution

        nodes = self._destroy.select_nodes(solution, self._num_rollouts)  # [*B, R, M]
        accept_kind, temperature = _acceptance_to_kernel(accept)
        nodes_to_remove = (
            nodes.reshape(-1, nodes.shape[-2], nodes.shape[-1])
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )
        match solution:
            case CVRPGraphSolution():
                return self._call_cvrp(solution, nodes_to_remove, accept_kind, temperature)
            case TSPGraphSolution():
                return self._call_tsp(solution, nodes_to_remove, accept_kind, temperature)
            case CMSTPSolution():
                return self._call_cmstp(solution, nodes_to_remove, accept_kind, temperature)
            case _:
                raise NotImplementedError(
                    f"NumbaSequentialImprovementStep not implemented for {type(solution)}"
                )

    def _call_cmstp(
        self,
        solution: CMSTPSolution,
        nodes_to_remove: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> CMSTPSolution:
        return solution.apply_and_repair_sequential(
            nodes_to_remove,
            n_repairs=self._n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=EPS,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_rollout_order,
        )

    def _call_cvrp(
        self,
        solution: CVRPGraphSolution,
        nodes_to_remove: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> CVRPGraphSolution:
        from core.lns.numba_kernels.cvrp_apply_and_repair import (
            apply_and_repair_sequential as cvrp_apply_and_repair_sequential,
        )
        from core.problems.cvrp.graph_solution import CVRPGraphSolution

        next_np, prev_np, locations, demand, capacity = solution.to_numpy_arrays()
        next_out, prev_out = cvrp_apply_and_repair_sequential(
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

    def _call_tsp(
        self,
        solution: TSPGraphSolution,
        nodes_to_remove: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> TSPGraphSolution:
        from core.lns.numba_kernels.tsp_apply_and_repair import (
            apply_and_repair_sequential as tsp_apply_and_repair_sequential,
        )
        from core.problems.tsp.graph_solution import TSPGraphSolution

        if self._insert_in_new_routes_only:
            raise NotImplementedError(
                "insert_in_new_routes_only is a CVRP-only concept (TSP has a single "
                "tour, no routes)."
            )

        next_np, prev_np, locations = solution.to_numpy_arrays()
        next_out, prev_out = tsp_apply_and_repair_sequential(
            next=next_np,
            prev=prev_np,
            nodes_to_remove=nodes_to_remove,
            locations=locations,
            n_repairs=self._n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=EPS,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_rollout_order,
        )
        return TSPGraphSolution.from_numpy_arrays(
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
