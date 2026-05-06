from collections.abc import Callable

from core.lns.operators.repair.greedy import CVRPSequentialGreedyNodeInsertionRepair
from core.lns.operators.repair.greedy_edge import TSPGreedyEdgeRepair
from core.models.nds.model import NDSModel
from core.models.rollout import Rollout
from core.problems.cvrp import CVRPSolution
from core.problems.solution import BaseSolution
from core.problems.tsp.graph_solution import TSPGraphSolution

ImprovementStep = Callable[[BaseSolution, NDSModel], BaseSolution]

ApplyRollout = Callable[[BaseSolution, Rollout], BaseSolution]

Repair = Callable[[BaseSolution, Rollout], BaseSolution]


def noop_improvement_step(
    solution: BaseSolution,
    model: NDSModel,
) -> BaseSolution:
    return solution


def apply_rollout(solution: BaseSolution, rollout: Rollout) -> BaseSolution:
    match solution:
        case CVRPSolution() as s:
            return s.remove_customers(rollout.actions)
        case TSPGraphSolution() as s:
            return s.remove_nodes(rollout.actions)
        case _:
            raise NotImplementedError(f"Not implemented for {type(solution)}")


def greedy_cvrp_repair(solution: CVRPSolution, rollout: Rollout) -> CVRPSolution:
    return CVRPSequentialGreedyNodeInsertionRepair()(solution)


def greedy_tsp_repair(solution: TSPGraphSolution, rollout: Rollout) -> TSPGraphSolution:
    return TSPGreedyEdgeRepair()(solution)


def only_expanding_apply_rollout(solution: BaseSolution, rollout: Rollout) -> BaseSolution:
    return solution.unsqueeze(1).expand(*solution.batch_size, rollout.log_p_of_actions.shape[1])


def noop_repair(
    solution: BaseSolution,
    rollout: Rollout,
) -> BaseSolution:
    return solution
