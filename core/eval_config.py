from __future__ import annotations

from pydantic import BaseModel, Field

from core.problems.config import ProblemParams


class EvaluationConfig(BaseModel):
    """
    Method-agnostic configuration for an evaluation run.
    """

    problem: ProblemParams = Field(default_factory=ProblemParams, description="Test problem")
    num_test_instances: int = Field(default=100, description="Number of test instances")
    test_seed: int = Field(default=2026, description="Sampler seed")
    num_plot_instances: int = Field(default=4, description="Number of instances to plot")
