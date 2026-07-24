from __future__ import annotations

from core.lns.operators.repair.base import RepairOperator
from core.problems.cmst.solution import CMSTSolution


class CMSTGreedyInsertRepair(RepairOperator[CMSTSolution]):
    """Registry entry for CMST greedy attach-repair.

    The actual deconstruct + greedy attach lives in the fused numba pass
    (:mod:`core.problems.cmst._numba`), used by both the training main loop
    (``NumbaApplyAndRepair``) and validation (``NumbaSequentialImprovementStep``). This
    operator exists so ``RepairOperator.create("cmst", "greedy_insert")`` resolves and
    ``ModelNodeDestroy`` can read ``requires_stitch``; the torch ``__call__`` is never
    invoked on the numba paths and is intentionally not implemented.
    """

    repair_type = "greedy_insert"
    problem_name = "cmst"
    requires_stitch = True

    def __init__(
        self,
        *,
        seed: int | None = None,
        deconstruction_size: int | None = None,
    ) -> None:
        self._seed = seed
        self._deconstruction_size = deconstruction_size

    def __call__(self, deconstructed_solution: CMSTSolution) -> CMSTSolution:
        raise NotImplementedError(
            "CMST repair runs through the numba apply-and-repair, not the torch operator."
        )
