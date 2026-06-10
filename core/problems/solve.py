from __future__ import annotations

from functools import singledispatch

from core.problems.instance import BaseProblemInstance
from core.problems.solution import BaseSolution


@singledispatch
def initial_solution(instance: BaseProblemInstance) -> BaseSolution:
    """Build the default start solution for a problem instance.

    Each problem registers its default via ``@initial_solution.register`` (see the
    per-problem ``solve`` modules). Dispatch is on the instance type.
    """
    raise NotImplementedError(
        f"No default start solution registered for {type(instance).__name__}."
    )
