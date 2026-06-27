from __future__ import annotations

from core.lns.utils import apply_nodes
from core.models.rollout import Rollout
from core.problems.solution import BaseSolution


def apply_rollout(
    solution: BaseSolution, rollout: Rollout, *, stitch: bool = False
) -> BaseSolution:
    return apply_nodes(solution, rollout.actions, stitch=stitch)
