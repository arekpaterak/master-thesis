from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.lns.apply_and_repair import NumbaApplyAndRepair
from core.lns.operators.repair.base import RepairOperator
from core.problems.base import Problem
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@dataclass(frozen=True)
class LNSComponents[I: BaseProblemInstance, S: BaseSolution]:
    problem: Problem[I, S]
    repairs: Mapping[str, type[RepairOperator[S]]]
    apply_and_repairs: Mapping[str, type[NumbaApplyAndRepair[S]]]
