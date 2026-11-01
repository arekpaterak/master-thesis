from __future__ import annotations

import numpy as np

from core.constants import EPS
from core.lns.apply_and_repair import NumbaApplyAndRepair
from core.problems.cmstp.lns import _numba
from core.problems.cmstp.lns.arrays import CMSTPInstanceArrays, CMSTPSolutionArrays


class CMSTPGreedyInsertApplyRepair(NumbaApplyAndRepair):
    """Fused numba greedy attach apply-and-repair for CMSTP."""

    def _apply_parallel(
        self, solution: CMSTPSolutionArrays, instance: CMSTPInstanceArrays, nodes: np.ndarray
    ) -> CMSTPSolutionArrays:
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
        solution: CMSTPSolutionArrays,
        instance: CMSTPInstanceArrays,
        nodes: np.ndarray,
        accept_kind: int,
        temperature: float,
    ) -> CMSTPSolutionArrays:
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
