from __future__ import annotations

from core.lns.components import LNSComponents
from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep, build_improvement_step
from core.models.nds.factory import NDSModel
from core.models.nds.metrics import RolloutMetricsRecorder


def build_nds_improvement_step(
    search: SearchParams,
    *,
    model: NDSModel,
    lns: LNSComponents,
    seed: int = 0,
    recorder: RolloutMetricsRecorder | None = None,
) -> ImprovementStep:
    from core.models.nds.destroy import ModelNodeDestroy

    destroy = ModelNodeDestroy(
        model=model,
        deconstruction_size=search.deconstruction_size,
        recorder=recorder,
    )
    return build_improvement_step(search, destroy=destroy, lns=lns, seed=seed)
