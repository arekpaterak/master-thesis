from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from core.problems import CVRPConfig, ProblemConfig


class SyntheticProblemSetting(BaseModel):
    """Randomly sampled (or dataset-file) instances of a fixed problem size.

    This is also the input contract for :class:`core.evaluation.Evaluator`.
    """

    model_config = {"extra": "forbid"}

    kind: Literal["synthetic"] = "synthetic"
    problem: ProblemConfig = Field(default_factory=CVRPConfig, description="Test problem")
    dataset_path: str | None = Field(
        default=None,
        description="If set, load instances from this file instead of sampling randomly.",
    )
    num_test_instances: int = Field(default=100, description="Number of test instances")
    batch_size: int = Field(default=100, description="Evaluation batch size")
    test_seed: int = Field(default=2026, description="Sampler seed")

    @property
    def label(self) -> str:
        return f"{self.problem.name}{self.problem.size}"

    @property
    def source(self) -> str:
        """Instance provenance: the dataset file for loaded instances, the seed otherwise."""
        if self.dataset_path is not None:
            return Path(self.dataset_path).stem
        return f"seed{self.test_seed}"


class CVRPLibProblemSetting(BaseModel):
    """A CVRPLIB benchmark set, filtered by instance size."""

    model_config = {"extra": "forbid"}

    kind: Literal["cvrplib"] = "cvrplib"
    set_name: str = Field(description="CVRPLIB set name, e.g. X.")
    min_customers: int = Field(default=0, description="Minimum customers (0 = no filter).")
    max_customers: int = Field(default=9999, description="Maximum customers (9999 = no filter).")

    @property
    def label(self) -> str:
        return f"cvrplib-{self.set_name}"


ProblemSetting = Annotated[
    SyntheticProblemSetting | CVRPLibProblemSetting, Field(discriminator="kind")
]
