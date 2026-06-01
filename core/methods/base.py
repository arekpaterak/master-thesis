from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.lns.acceptance_criterion import AcceptanceCriterion
from core.lns.config import SearchParams
from core.lns.improvement_step import ImprovementStep
from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@dataclass(frozen=True)
class Method:
    """A built evaluation method, ready to be plugged into the eval loop.

    Concrete methods (e.g. NDSTrainedMethod, RandomDestroyMethod) live under
    ``core.methods`` and expose a classmethod ``build(config, ...)`` that
    constructs an instance of this dataclass.
    """

    name: str
    search: SearchParams
    generate_start_solution: Callable[[BaseProblemInstance], BaseSolution]
    improvement_step: ImprovementStep
    criterion: AcceptanceCriterion
