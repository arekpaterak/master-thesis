from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for evaluating a trained NDS model on a test set."""

    num_test_instances: int = Field(default=100, description="Test instances to evaluate.")
    test_seed: int = Field(default=2026, description="Sampler seed for test instances.")
    num_improvement_steps: int = Field(default=200, description="LNS iterations per instance.")
    rollouts_per_step: int = Field(default=100, description="Rollouts per LNS step.")
    n_repairs: int = Field(default=5, description="Repair attempts per rollout; best is selected.")
    temperature_start: float = Field(default=0.1, description="SA start temperature.")
    temperature_min: float = Field(default=0.001, description="SA final temperature.")
    improvement_step_type: Literal["sequential", "parallel"] = Field(
        default="sequential", description="Model-driven improvement step variant."
    )
    num_plot_instances: int = Field(default=4, description="Test instances to render PDFs for.")
    method_suffix: str | None = Field(
        default=None, description="Optional suffix appended to the method column."
    )
