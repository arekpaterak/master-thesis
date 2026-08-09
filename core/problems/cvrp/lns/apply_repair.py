from __future__ import annotations

import numpy as np

from core.constants import EPS
from core.lns.apply_and_repair import NumbaApplyAndRepair
from core.problems.cvrp.lns import _numba
from core.problems.cvrp.lns.arrays import CVRPInstanceArrays, CVRPSolutionArrays


class CVRPGreedyInsertApplyRepair(NumbaApplyAndRepair):
    """Fused numba greedy cheapest-insertion apply-and-repair for CVRP."""

    problem_name = "cvrp"
    repair_name = "greedy_insert"
    supports_affected_routes_only = True

    def _apply_parallel(
        self, solution: CVRPSolutionArrays, instance: CVRPInstanceArrays, nodes: np.ndarray
    ) -> CVRPSolutionArrays:
        out, _objective = _numba.apply_and_repair_parallel(
            solution,
            instance,
            nodes,
            n_repairs=self._n_repairs,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_order,
            insert_in_affected_routes_only=self._insert_in_affected_routes_only,
        )
        return out

    def _apply_sequential(
        self,
        solution: CVRPSolutionArrays,
        instance: CVRPInstanceArrays,
        nodes: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> CVRPSolutionArrays:
        return _numba.apply_and_repair_sequential(
            solution,
            instance,
            nodes,
            n_repairs=self._n_repairs,
            accept_kind=accept_kind,
            temperature=temperature,
            eps=EPS,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_order,
            insert_in_affected_routes_only=self._insert_in_affected_routes_only,
        )
