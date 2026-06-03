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
            return s.remove_customers(rollout.actions)
        case CVRPGraphSolution() as s:
            return s.remove_customers(rollout.actions, stitch=stitch)
        case TSPGraphSolution() as s:
            return s.remove_nodes(rollout.actions)
        case _:
            raise NotImplementedError(f"Not implemented for {type(solution)}")


def only_expanding_apply_rollout(solution: BaseSolution, rollout: Rollout, **_) -> BaseSolution:
    """
    Only expands the ``solution`` from (B, destruction_size) to (B, num_rollouts, destruction_size), based on the ``rollout`` dimensionality.
    """
    return solution.unsqueeze(1).expand(*solution.batch_size, rollout.batch_size[1])
