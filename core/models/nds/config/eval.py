from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.nds.config.common import ProblemParams, SearchParams


class EvaluationConfig(BaseModel):
    """Configuration for evaluating a trained NDS model on a test set."""

    problem: ProblemParams = Field(default_factory=ProblemParams, description="Test problem")
    num_test_instances: int = Field(default=100, description="Number of test instances")
    test_seed: int = Field(default=2026, description="Sampler seed")
    search: SearchParams = Field(
        default_factory=lambda: SearchParams(
            num_improvement_steps=200, num_rollouts=100, n_repairs=5
        ),
        description="LNS search parameters",
    )
    num_plot_instances: int = Field(default=4, description="Number of instances to plot")
    method_suffix: str | None = Field(default=None, description="Optional suffix for method name")
