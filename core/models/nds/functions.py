from collections.abc import Callable

from core.lns.operators.repair.greedy import GreedyCVRPRepair
from core.models.nds.model import NDSModel
from core.models.rollout import Rollout
from core.problems.cvrp import CVRPSolution
from core.problems.solution import BaseSolution

ImprovementStep = Callable[[BaseSolution, NDSModel], BaseSolution]

ApplyRollout = Callable[[BaseSolution, Rollout], BaseSolution]

Repair = Callable[[BaseSolution, Rollout], BaseSolution]


def noop_improvement_step(
    solution: BaseSolution,
    model: NDSModel,
) -> BaseSolution:
    return solution


def apply_rollout(
    solution: CVRPSolution,
    rollout: Rollout,
) -> CVRPSolution:
    return solution.remove_customers(rollout.actions)


def only_expanding_apply_rollout(solution: BaseSolution, rollout: Rollout) -> BaseSolution:
    return solution.unsqueeze(1).expand(*solution.batch_size, rollout.log_p_of_actions.shape[1])


def noop_repair(
    solution: BaseSolution,
    rollout: Rollout,
) -> BaseSolution:
    return solution


def greedy_cvrp_repair(solution: CVRPSolution, rollout: Rollout) -> CVRPSolution:
    return GreedyCVRPRepair()(solution)
