from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep
from core.lns.start_solution import with_warmup
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution

if TYPE_CHECKING:
    from core.lns.components import LNSComponents
    from core.problems import ProblemConfig


@dataclass(frozen=True)
class Method:
    search: SearchParams
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution]
    improvement_step: ImprovementStep

    @classmethod
    def from_lns(
        cls, search: SearchParams, *, lns: LNSComponents, improvement_step: ImprovementStep
    ) -> Method:
        return cls(
            search=search,
            generate_start_solution=with_warmup(
                lns.problem.initial_solution, lns=lns, search=search
            ),
            improvement_step=improvement_step,
        )


class MethodConfigBase(BaseModel):
    @abstractmethod
    def build(self, *, problem: ProblemConfig, device: torch.device) -> Method: ...
