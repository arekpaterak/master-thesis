from __future__ import annotations

from core.lns.config import SearchParams
from core.lns.improvement_step import (
    ImprovementStep,
    NumbaSequentialImprovementStep,
    ParallelImprovementStep,
    SequentialImprovementStep,
)
from core.lns.operators.repair.base import RepairOperator
from core.models.nds.factory import NDSModel


def build_nds_improvement_step(
    search: SearchParams,
    *,
    model: NDSModel,
    repair: RepairOperator,
    seed: int = 0,
) -> ImprovementStep:
    """Create a :class:`ModelNodeDestroy` and wire it to the appropriate LNS improvement step."""
    from core.models.nds.destroy import ModelNodeDestroy

    destroy = ModelNodeDestroy(
        model=model,
        deconstruction_size=search.deconstruction_size,
        stitch=repair.requires_stitch,
    )
    match search.improvement_step_type:
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
        case "numba_sequential":
            return NumbaSequentialImprovementStep(
                destroy=destroy,
                repair=repair,
                num_rollouts=search.num_rollouts,
                n_repairs=search.n_repairs,
                seed=seed,
                first_repair_uses_rollout_order=search.first_repair_uses_rollout_order,
                insert_in_new_routes_only=search.insert_in_new_routes_only,
            )
        case _:
            raise ValueError(f"Unknown improvement_step_type: {search.improvement_step_type!r}")
