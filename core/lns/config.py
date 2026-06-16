from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    SimulatedAnnealingAcceptanceConfig,
)

RepairType = Literal["greedy_reconnect", "greedy_insert", "noop"]


# TODO: Add field titles, and use them for widgets
class SearchParams(BaseModel):
    """
    Configuration for an LNS-style search (destroy / repair / accept loop).
    """

    repair_type: RepairType = Field(default="greedy_reconnect", description="Repair type.")
    num_improvement_steps: int = Field(default=10, description="Number of search iterations.")
    num_rollouts: int = Field(default=128, description="Rollouts per step.")
    deconstruction_size: int = Field(default=15, description="Number of nodes removed per rollout.")
    n_repairs: int = Field(default=5, description="Repair attempts per rollout; best is kept.")
    improvement_step_type: Literal["sequential", "parallel", "numba_sequential"] = Field(
        default="parallel",
        description="Improvement step variant.",
    )
    acceptance: AcceptanceCriterionConfig = Field(
        default_factory=SimulatedAnnealingAcceptanceConfig,
        description="Acceptance criterion (greedy or simulated annealing).",
    )
    n_augmentations: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of D4 augmentations per instance (1 = no augmentation).",
    )
    augmented_threshold_factor: float = Field(
        default=15.0,
    )
