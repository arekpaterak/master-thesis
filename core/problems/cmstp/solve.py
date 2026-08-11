from __future__ import annotations

from core.problems.cmstp.construction import star
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.cmstp.solution import CMSTPSolution
from core.problems.solve import generate_initial_solution


@generate_initial_solution.register
def _(instance: CMSTPInstance) -> CMSTPSolution:
    return star(instance)
