from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    SimulatedAnnealingAcceptanceConfig,
)

RepairType = Literal["greedy_reconnect", "greedy_insert", "noop"]


class SearchParams(BaseModel):
    """
    Configuration for an LNS-style search (destroy / repair / accept loop).
    """

    repair_type: RepairType = Field(
        default="greedy_reconnect", title="Repair type", description="Repair type."
    )
    num_improvement_steps: int = Field(
        default=10, title="Improvement steps", description="Number of search iterations."
    )
    num_rollouts: int = Field(
        default=128, title="Rollouts per instance", description="Rollouts per step."
    )
    deconstruction_size: int = Field(
        default=15, title="Deconstruction size", description="Number of nodes removed per rollout."
    )
    n_repairs: int = Field(
        default=5, title="Repair attempts", description="Repair attempts per rollout; best is kept."
    )
    first_repair_uses_rollout_order: bool = Field(
        default=False,
        title="First repair uses rollout order",
        description=(
            "If set, the first of the n_repairs greedy insertions follows the model "
            "rollout removal order instead of a random order. Only supported by the "
            "numba_sequential improvement step."
        ),
    )
    improvement_step_type: Literal["sequential", "parallel", "numba_sequential"] = Field(
        default="parallel",
        title="Improvement step type",
        description="Improvement step variant.",
    )
    insert_in_new_routes_only: bool = Field(
        default=False,
        title="Insert in new routes only",
        description=(
            "If set, removed customers may only be reinserted into routes the destroy "
            "touched (their remnants) or routes created during this repair, never an "
            "untouched route. Only supported by the numba_sequential improvement step."
        ),
    )
    acceptance: AcceptanceCriterionConfig = Field(
        default_factory=SimulatedAnnealingAcceptanceConfig,
        description="Acceptance criterion (greedy or simulated annealing).",
    )
    n_augmentations: int = Field(
        default=1,
        ge=1,
        le=8,
        title="Augmentations",
        description="Number of D4 augmentations per instance (1 = no augmentation).",
    )
    augmented_threshold_factor: float = Field(
        default=15.0,
        title="Augmentation threshold factor",
    )
