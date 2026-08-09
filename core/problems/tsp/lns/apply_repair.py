from __future__ import annotations

import numpy as np

from core.constants import EPS
from core.lns.apply_and_repair import NumbaApplyAndRepair
from core.problems.tsp.lns import _numba
from core.problems.tsp.lns.arrays import TSPInstanceArrays, TSPSolutionArrays


class TSPGreedyInsertApplyRepair(NumbaApplyAndRepair):
    """Fused numba greedy cheapest-insertion apply-and-repair for TSP."""

    problem_name = "tsp"
    repair_name = "greedy_insert"

    def _apply_parallel(
        self, solution: TSPSolutionArrays, instance: TSPInstanceArrays, nodes: np.ndarray
    ) -> TSPSolutionArrays:
        out, _objective = _numba.apply_and_repair_parallel(
            solution,
            instance,
            nodes,
            n_repairs=self._n_repairs,
            base_seed=self._seed,
            first_repair_uses_order=self._first_repair_uses_order,
        )
        return out

    def _apply_sequential(
        self,
        solution: TSPSolutionArrays,
        instance: TSPInstanceArrays,
        nodes: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> TSPSolutionArrays:
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
        )
