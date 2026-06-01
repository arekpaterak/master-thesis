from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    SimulatedAnnealingAcceptanceConfig,
)


class SearchParams(BaseModel):
    """
    Configuration for an LNS-style search (destroy / repair / accept loop).
    """

    num_improvement_steps: int = Field(default=10, description="Number of search iterations.")
    num_rollouts: int = Field(default=128, description="Rollouts per step.")
    deconstruction_size: int = Field(default=15, description="Number of nodes removed per rollout.")
    n_repairs: int = Field(default=1, description="Repair attempts per rollout; best is kept.")
    improvement_step_type: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description="Improvement step variant.",
    )
    acceptance: AcceptanceCriterionConfig = Field(
        default_factory=SimulatedAnnealingAcceptanceConfig,
        description="Acceptance criterion (greedy or simulated annealing).",
    )
