from __future__ import annotations

from core.models.rollout import Rollout
from core.problems.cvrp import CVRPSolution
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution


# TODO: Refactor
def apply_rollout(
    solution: BaseSolution, rollout: Rollout, *, stitch: bool = False
) -> BaseSolution:
    match solution:
        case CVRPSolution() as s:
            return s.remove_nodes(rollout.actions)
        case CVRPGraphSolution() as s:
            return s.remove_nodes(rollout.actions, stitch=stitch)
        case TSPGraphSolution() as s:
            return s.remove_nodes(rollout.actions)
        case _:
            raise NotImplementedError(f"Not implemented for {type(solution)}")


def only_expanding_apply_rollout(solution: BaseSolution, rollout: Rollout, **_) -> BaseSolution:
    """
    Only expands the ``solution`` by adding the rollout dim as the innermost batch
    dim, e.g. ``(B, ...) -> (B, num_rollouts, ...)`` or ``(B, A, ...) -> (B, A, num_rollouts, ...)``,
    based on the ``rollout`` dimensionality.
    """
    num_rollouts = rollout.batch_size[-1]
    return solution.unsqueeze(-1).expand(*solution.batch_size, num_rollouts)
