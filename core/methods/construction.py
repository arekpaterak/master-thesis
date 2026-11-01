from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field

from core.lns.acceptance_criterion import UnconditionalAcceptanceConfig
from core.lns.config import SearchParams
from core.lns.improvement_step import NoopImprovementStep
from core.methods.base import Method, MethodConfigBase
from core.problems import ProblemConfig, registry


class ConstructionMethodConfig(MethodConfigBase):
    type: Literal["construction"] = "construction"
    construction_name: str = Field(description="Construction heuristic name.")

    def build(self, *, problem: ProblemConfig, device: torch.device) -> Method:
        heuristic = registry.problem(problem.name).construction(self.construction_name)
        # No improvement search: 0 steps, and a criterion that needs no step schedule.
        search = SearchParams(num_improvement_steps=0, acceptance=UnconditionalAcceptanceConfig())
        return Method(
            search=search,
            generate_start_solution=heuristic,
            improvement_step=NoopImprovementStep(),
        )
