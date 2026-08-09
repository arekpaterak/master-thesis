from __future__ import annotations

import torch

from core.lns.operators.repair import RepairOperator
from core.problems.cvrp import CVRPSolution
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution


def remove_nodes(
    solution: BaseSolution, nodes: torch.Tensor, *, stitch: bool = False
) -> BaseSolution:
    """Remove ``nodes`` from ``solution``.

    ``nodes`` is [*B, M] for a single set or [*B, R, M] for R parallel sets,
    where *B matches the leading dims of ``solution.batch_size``.
    """
    match solution:
        case CVRPSolution() as s:
            return s.remove_nodes(nodes)
        case CVRPGraphSolution() as s:
            return s.remove_nodes(nodes, stitch=stitch)
        case TSPGraphSolution() as s:
            return s.remove_nodes(nodes, stitch=stitch)
        case _:
            raise NotImplementedError(f"remove_nodes not implemented for {type(solution)}")


def best_repair(repair: RepairOperator, deconstructed: BaseSolution, n: int) -> BaseSolution:
    """Repair ``deconstructed`` solution ``n`` times and return the best by objective."""
    best = repair(deconstructed)
    for _ in range(n - 1):
        candidate = repair(deconstructed)
        best = candidate.where(candidate.objective() < best.objective(), best)
    return best
