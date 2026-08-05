from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field

from core.eval.problem_setting import (
    CVRPLibProblemSetting,
    ProblemSetting,
)
from core.logging_config import LoggingConfig
from core.methods import MethodConfig


class EvaluationRunConfig(BaseModel):
    """A single method evaluated on a single (grouped) setting -- one eval run."""

    model_config = {"extra": "forbid"}

    name: str
    group: str
    method: MethodConfig
    setting: ProblemSetting
    num_plot_instances: int = 0
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def problem_name(self) -> str:
        if isinstance(self.setting, CVRPLibProblemSetting):
            return "cvrp"
        return self.setting.problem.name


class EvaluationConfig(BaseModel):
    """A method evaluated across settings, grouped by role (e.g. main, generalisation)."""

    model_config = {"extra": "forbid"}

    name: str | None = Field(default=None, description="Config name (defaults to the filename).")
    method: MethodConfig = Field(description="The method under evaluation.")
    settings: dict[str, list[ProblemSetting]] = Field(
        description="Settings grouped by role, e.g. {main: [...], generalisation: [...]}.",
    )
    num_plot_instances: int = Field(
        default=0, description="Instances to plot per synthetic setting."
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def problem_name(self) -> str:
        names = {
            "cvrp" if isinstance(setting, CVRPLibProblemSetting) else setting.problem.name
            for group in self.settings.values()
            for setting in group
        }
        if len(names) > 1:
            raise ValueError(f"All settings must share one problem type, got {names}.")
        return names.pop() if names else "cvrp"

    def runs(self) -> list[EvaluationRunConfig]:
        """One :class:`EvaluationRunConfig` per (group, setting), in declaration order."""
        return [
            EvaluationRunConfig(
                name=self.name or "",
                group=group,
                method=self.method,
                setting=setting,
                num_plot_instances=self.num_plot_instances,
                logging=self.logging,
            )
            for group, settings in self.settings.items()
            for setting in settings
        ]

    @classmethod
    def from_file(cls, path: str | Path, *, root: Path | None = None) -> Self:
        path = Path(path)
        root = root or path.parent
        raw = yaml.safe_load(path.read_text()) or {}
        raw.setdefault("name", path.stem)

        method = raw.get("method")
        if isinstance(method, dict) and "search" in method:
            method["search"] = cls._resolve(method["search"], root)
        if isinstance(raw.get("settings"), dict):
            raw["settings"] = {
                group: [cls._resolve(setting, root) for setting in settings]
                for group, settings in raw["settings"].items()
            }

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
