from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    SimulatedAnnealingAcceptanceConfig,
)

RepairType = Literal["greedy_reconnect", "greedy_insert"]


class StartSolutionParams(BaseModel):
    """Random-improvement warmup applied to the initial solution.

    Only the warmup-specific overrides live here; the repair reuses the parent search's
    ``repair_type`` and ``improvement_step_type``. The warmup itself is a single-rollout,
    single-repair.
    """

    num_improvement_steps: int = Field(
        default=50,
        title="Warmup iterations",
        description="Warmup random-destroy/recreate iterations on the initial solution. 0 disables.",
    )
    deconstruction_size: int = Field(
        default=15,
        title="Warmup deconstruction size",
        description="Number of customers removed per warmup iteration.",
    )
    destroy_type: Literal["random_node"] = Field(
        default="random_node",
        title="Warmup destroy",
        description="Destroy operator used for the warmup.",
    )


class SearchParams(BaseModel):
    """
    Configuration for an LNS-style search (destroy / repair / accept loop).
    """

    model_config = ConfigDict(populate_by_name=True)

    repair_type: RepairType = Field(
        default="greedy_insert", title="Repair type", description="Repair type."
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
    improvement_step_type: Literal[
        "sequential", "parallel", "numba_sequential", "numba_parallel"
    ] = Field(
        default="parallel",
        title="Improvement step type",
        description="Improvement step variant. torch (sequential/parallel) or numba backend.",
    )
    apply_and_repair_backend: Literal["torch", "numba"] = Field(
        default="numba",
        title="Apply-and-repair backend",
        description=("Backend for the training deconstruct+repair of each rollout."),
    )
    insert_in_affected_routes_only: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "insert_in_affected_routes_only", "insert_in_new_routes_only"
        ),
        title="Insert in affected routes only",
        description=(
            "If set, removed customers may only be reinserted into routes the destroy "
            "touched (their remnants) or routes created during this repair, never an "
            "untouched route. Only supported by the numba_sequential improvement step, "
            "and only for CVRP (TSP and CMSTP reject it)."
        ),
    )
    acceptance: AcceptanceCriterionConfig = Field(
        default_factory=SimulatedAnnealingAcceptanceConfig,
        description="Acceptance criterion (greedy or simulated annealing).",
    )
    start_solution_params: StartSolutionParams = Field(
        default_factory=StartSolutionParams,
        description="Random-improvement warmup applied to the initial solution before search.",
    )
    n_augmentations: int = Field(
        default=1,
        ge=1,
        le=8,
        title="Augmentations",
        description="Number of D4 augmentations per instance (1 = no augmentation).",
    )
