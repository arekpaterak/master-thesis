from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

from core.lns.config import SearchParams
from core.lns.operators.repair.base import RepairOperator
from core.lns.utils import best_repair
from core.models.nds.apply_rollout import apply_rollout
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from core.problems.cvrp.graph_solution import CVRPGraphSolution
    from core.problems.tsp.graph_solution import TSPGraphSolution


class ApplyAndRepair(ABC):
    """Apply a batch of rollouts to a solution and repair each into candidate solutions.

    Given a solution (batch ``(B,)``) and a rollout with ``actions`` shaped ``(B, K, M)``,
    deconstructs the solution by each rollout's node set and greedily reconstructs it,
    returning the repaired candidates batched ``(B, K)``. No acceptance -- the caller
    (training) scores the candidates. This is the parallel counterpart to a single
    ``ParallelImprovementStep``, minus the accept/reduce.
    """

    @abstractmethod
    def __call__(self, solution: BaseSolution, rollout: Rollout) -> BaseSolution: ...


class TorchApplyAndRepair(ApplyAndRepair):
    """Vectorised torch backend: apply_rollout (destroy) then best-of-n greedy repair.

    This is the original NDS training path. The repair materialises the full pairwise
    distance matrix; prefer :class:`NumbaApplyAndRepair` to avoid that memory spike.
    """

    def __init__(self, *, repair: RepairOperator, n_repairs: int, stitch: bool) -> None:
        self._repair = repair
        self._n_repairs = n_repairs
        self._stitch = stitch

    def __call__(self, solution: BaseSolution, rollout: Rollout) -> BaseSolution:
        deconstructed = apply_rollout(solution, rollout, stitch=self._stitch)
        return best_repair(self._repair, deconstructed, self._n_repairs)


class NumbaApplyAndRepair(ApplyAndRepair):
    """CPU/numba backend: the kernel deconstructs and best-of-n greedy repairs each rollout.

    Distances are computed on the fly in the kernel, so the ``(B, K, N+1, N+1)`` distance
    matrix the torch repair materialises never exists. The kernel reimplements the greedy
    insertion, so no ``RepairOperator`` is needed here; ``first_repair_uses_rollout_order``
    and ``insert_in_new_routes_only`` mirror the numba search repair (unlike the torch
    backend, whose ``CVRPGreedyInsertRepair`` ignores both).
    """

    def __init__(
        self,
        *,
        n_repairs: int,
        first_repair_uses_rollout_order: bool = False,
        insert_in_new_routes_only: bool = False,
        seed: int = 0,
    ) -> None:
        self._n_repairs = n_repairs
        self._first_repair_uses_rollout_order = first_repair_uses_rollout_order
        self._insert_in_new_routes_only = insert_in_new_routes_only
        self._seed = seed

    def __call__(self, solution: BaseSolution, rollout: Rollout) -> BaseSolution:
        from core.problems.cvrp.graph_solution import CVRPGraphSolution
        from core.problems.tsp.graph_solution import TSPGraphSolution

        match solution:
            case CVRPGraphSolution():
                return self._call_cvrp(solution, rollout)
            case TSPGraphSolution():
                return self._call_tsp(solution, rollout)
            case _:
                raise NotImplementedError(
                    f"NumbaApplyAndRepair not implemented for {type(solution)}"
                )

    def _call_cvrp(self, solution: CVRPGraphSolution, rollout: Rollout) -> CVRPGraphSolution:
        from core.lns.numba_kernels.cvrp_apply_and_repair import apply_and_repair_parallel
        from core.problems.cvrp.graph_solution import CVRPGraphSolution

        next_np, prev_np, locations, demand, capacity = solution.to_numpy_arrays()
        nodes = self._nodes_to_numpy(rollout, num_instances=next_np.shape[0])
        next_out, prev_out, _ = apply_and_repair_parallel(
            next=next_np,
            prev=prev_np,
            nodes=nodes,
            locations=locations,
            demand=demand,
            capacity=capacity,
            n_repairs=self._n_repairs,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_rollout_order,
            insert_in_new_routes_only=self._insert_in_new_routes_only,
        )
        return self._rebuild(CVRPGraphSolution, solution, next_out, prev_out)

    def _call_tsp(self, solution: TSPGraphSolution, rollout: Rollout) -> TSPGraphSolution:
        from core.lns.numba_kernels.tsp_apply_and_repair import apply_and_repair_parallel
        from core.problems.tsp.graph_solution import TSPGraphSolution

        if self._insert_in_new_routes_only:
            raise NotImplementedError(
                "insert_in_new_routes_only is a CVRP-only concept (TSP has a single "
                "tour, no routes)."
            )

        next_np, prev_np, locations = solution.to_numpy_arrays()
        nodes = self._nodes_to_numpy(rollout, num_instances=next_np.shape[0])
        next_out, prev_out, _ = apply_and_repair_parallel(
            next=next_np,
            prev=prev_np,
            nodes=nodes,
            locations=locations,
            n_repairs=self._n_repairs,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_rollout_order,
        )
        return self._rebuild(TSPGraphSolution, solution, next_out, prev_out)

    @staticmethod
    def _nodes_to_numpy(rollout: Rollout, *, num_instances: int) -> np.ndarray:
        num_rollouts, deconstruction_size = rollout.actions.shape[-2:]
        return (
            rollout.actions.reshape(num_instances, num_rollouts, deconstruction_size)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int64, copy=False)
        )

    @staticmethod
    def _rebuild(
        solution_cls: type[BaseSolution],
        solution: BaseSolution,
        next_out: np.ndarray,
        prev_out: np.ndarray,
    ) -> BaseSolution:
        num_rollouts = next_out.shape[1]
        # The instance is shared across the K rollouts; expand a view (not the N^2
        # distance matrix) so the (B, K) candidate solution can score its objective.
        instance = solution.instance.reshape(*solution.batch_size, 1).expand(
            *solution.batch_size, num_rollouts
        )
        return solution_cls.from_numpy_arrays(
            instance=instance,
            next=next_out,
            prev=prev_out,
            batch_size=torch.Size([*solution.batch_size, num_rollouts]),
            device=solution.device,
        )


def build_apply_and_repair(
    search: SearchParams, *, repair: RepairOperator, seed: int = 0
) -> ApplyAndRepair:
    match search.apply_and_repair_backend:
        case "torch":
            return TorchApplyAndRepair(
                repair=repair, n_repairs=search.n_repairs, stitch=repair.requires_stitch
            )
        case "numba":
            _validate_numba_repair(repair)
            return NumbaApplyAndRepair(
                n_repairs=search.n_repairs,
                first_repair_uses_rollout_order=search.first_repair_uses_rollout_order,
                insert_in_new_routes_only=search.insert_in_new_routes_only,
                seed=seed,
            )


def _validate_numba_repair(repair: RepairOperator) -> None:
    match getattr(repair, "problem_name", None):
        case "cvrp" if repair.repair_type == "greedy_insert" and repair.requires_stitch:
            return
        case "tsp" if repair.repair_type == "greedy_insert":
            return
    raise NotImplementedError(
        "NumbaApplyAndRepair only supports the stitched CVRP greedy-insert repair or the "
        f"TSP greedy-insert repair, got {type(repair).__name__} "
        f"(problem_name={getattr(repair, 'problem_name', None)!r})."
    )
