from __future__ import annotations

from core.problems.cmst.instance import CMSTInstance
from core.problems.cmst.solution import CMSTSolution
from core.problems.solve import generate_initial_solution


@generate_initial_solution.register
def _(instance: CMSTInstance) -> CMSTSolution:
    return CMSTSolution.star(instance)
