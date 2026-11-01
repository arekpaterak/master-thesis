from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field

from core.lns.config import SearchParams
from core.lns.improvement_step import build_random_node_destroy_step
from core.methods.base import Method, MethodConfigBase
from core.problems import ProblemConfig, registry


class RandomDestroyMethodConfig(MethodConfigBase):
    type: Literal["random-destroy"] = "random-destroy"
    search: SearchParams = Field(default_factory=SearchParams)
    seed: int | None = Field(default=None, description="Optional seed for the random destroy.")

    def build(self, *, problem: ProblemConfig, device: torch.device) -> Method:
        lns = registry.lns(problem.name)
        improvement_step = build_random_node_destroy_step(self.search, lns=lns, seed=self.seed)
        return Method.from_lns(self.search, lns=lns, improvement_step=improvement_step)


RANDOM_DESTROY_PRESETS: dict[str, RandomDestroyMethodConfig] = {
    "source": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_insert",
            num_rollouts=1,
            deconstruction_size=15,
            n_repairs=5,
            improvement_step_type="sequential",
        )
    ),
    "parallel": RandomDestroyMethodConfig(
        search=SearchParams(
            repair_type="greedy_insert",
            num_rollouts=100,
            deconstruction_size=15,
            n_repairs=1,
            improvement_step_type="parallel",
        )
    ),
}
