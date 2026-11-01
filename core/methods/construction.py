from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import UnconditionalAcceptanceConfig
from core.lns.config import SearchParams
from core.lns.improvement_step import NoopImprovementStep
from core.methods.base import Method
from core.problems.base import Problem


class ConstructionMethodConfig(BaseModel):
    type: Literal["construction"] = "construction"
    construction_name: str = Field(description="Construction heuristic name.")


class ConstructionMethod:
    @classmethod
    def build(cls, config: ConstructionMethodConfig, *, problem: Problem) -> Method:
        heuristic = problem.construction(config.construction_name)
        # No improvement search: 0 steps, and a criterion that needs no step schedule.
        search = SearchParams(num_improvement_steps=0, acceptance=UnconditionalAcceptanceConfig())
        return Method(
            search=search,
            generate_start_solution=heuristic,
            improvement_step=NoopImprovementStep(),
        )
