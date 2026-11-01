from __future__ import annotations

from core.lns.operators.repair import RepairOperator
from core.problems.solution import BaseSolution


def best_repair(repair: RepairOperator, deconstructed: BaseSolution, n: int) -> BaseSolution:
    """Repair ``deconstructed`` solution ``n`` times and return the best by objective."""
    best = repair(deconstructed)
    for _ in range(n - 1):
        candidate = repair(deconstructed)
        best = candidate.where(candidate.objective() < best.objective(), best)
    return best
