from __future__ import annotations

from pydantic import BaseModel, Field

from core.eval.problem_setting import CVRPLibProblemSetting, ProblemSetting
from core.logging_config import LoggingConfig
from core.methods import MethodConfig


class EvaluationConfig(BaseModel):
    """A method evaluated across a list of problem settings."""

    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, description="Config name (defaults to the filename).")
    method: MethodConfig = Field(description="The method under evaluation.")
    problems: list[ProblemSetting] = Field(description="Problem settings to evaluate on.")
    num_plot_instances: int = Field(
        default=0, description="Instances to plot per synthetic setting."
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def problem_name(self) -> str:
        """The single problem type shared by all settings (CVRPLIB is always cvrp)."""
        names = {
            "cvrp" if isinstance(setting, CVRPLibProblemSetting) else setting.problem.name
            for setting in self.problems
        }
        if len(names) > 1:
            raise ValueError(f"All problem settings must share one problem type, got {names}.")
        return names.pop() if names else "cvrp"
