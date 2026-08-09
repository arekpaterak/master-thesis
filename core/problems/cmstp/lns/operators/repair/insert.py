from __future__ import annotations

from core.lns.operators.repair.base import RepairOperator
from core.problems.cmstp.solution import CMSTPSolution


class CMSTPGreedyInsertRepair(RepairOperator[CMSTPSolution]):
    repair_type = "greedy_insert"
    problem_name = "cmstp"
    requires_stitch = True

    def __init__(
        self,
        *,
        seed: int | None = None,
        deconstruction_size: int | None = None,
    ) -> None:
        self._seed = seed
        self._deconstruction_size = deconstruction_size

    def __call__(self, deconstructed_solution: CMSTPSolution) -> CMSTPSolution:
        raise NotImplementedError(
            "CMSTP repair runs through the numba apply-and-repair, not the torch operator."
        )
