from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProblemParams(BaseModel):
    """Configuration for the optimisation problem."""

    name: str = Field(default="cvrp", description="Problem to solve.")
    size: int = Field(default=20, description="Number of customers.")


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
        description="Model-driven improvement step variant.",
    )
    temperature_start: float = Field(
        default=0.1, description="SA initial temperature (unused for greedy acceptance)."
    )
    temperature_min: float = Field(
        default=0.001, description="SA minimum temperature (unused for greedy acceptance)."
    )
