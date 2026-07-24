from __future__ import annotations

from core.problems.cmst.instance import CMSTInstance
from core.problems.cmst.solution import CMSTSolution
from core.problems.solve import generate_initial_solution


@generate_initial_solution.register
def _(instance: CMSTInstance) -> CMSTSolution:
    """Default CMST start: the star tree (every terminal a direct child of the root)."""
    return CMSTSolution.star(instance)
