from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
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
        names = {
            "cvrp" if isinstance(setting, CVRPLibProblemSetting) else setting.problem.name
            for setting in self.problems
        }
        if len(names) > 1:
            raise ValueError(f"All problem settings must share one problem type, got {names}.")
        return names.pop() if names else "cvrp"

    @classmethod
    def from_file(cls, path: str | Path, *, root: Path | None = None) -> Self:
        path = Path(path)
        root = root or path.parent
        raw = yaml.safe_load(path.read_text()) or {}
        raw.setdefault("name", path.stem)

        method = raw.get("method")
        if isinstance(method, dict) and "search" in method:
            method["search"] = cls._resolve(method["search"], root)
        if "problems" in raw:
            raw["problems"] = [cls._resolve(problem, root) for problem in raw["problems"]]

        return cls.model_validate(raw)

    @staticmethod
    def _resolve(value: object, root: Path) -> object:
        if isinstance(value, str):
            return yaml.safe_load((root / value).read_text()) or {}
        return value

    def write_yaml(self, *, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return path
