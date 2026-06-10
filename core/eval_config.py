from __future__ import annotations

from pydantic import BaseModel, Field

from core.problems import CVRPConfig, ProblemConfig


class EvaluationConfig(BaseModel):
    """
    Method-agnostic configuration for an evaluation run.
    """

    model_config = {"extra": "forbid"}

    problem: ProblemConfig = Field(default_factory=CVRPConfig, description="Test problem")
    num_test_instances: int = Field(default=100, description="Number of test instances")
    batch_size: int = Field(
        default=100, description="Evaluation batch size (last batch may be short)"
    )
    test_seed: int = Field(default=2026, description="Sampler seed")
    num_plot_instances: int = Field(default=4, description="Number of instances to plot")
    wandb_project: str = Field(
        default="nds",
        description="If non-empty, eval results + plots are uploaded under this wandb project.",
    )
