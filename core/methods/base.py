from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@dataclass(frozen=True)
class Method:
    name: str
    search: SearchParams
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution]
    improvement_step: ImprovementStep
