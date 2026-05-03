from collections.abc import Callable

from core.lns.operators.repair.greedy import GreedyCVRPRepair
from core.models.nds.model import NDSModel
from core.models.nds.rollout import NDSRollout
from core.problems.cvrp import CVRPSolution
from core.problems.solution import BaseSolution

ImprovementStep = Callable[[BaseSolution, NDSModel], BaseSolution]

ApplyRollout = Callable[[BaseSolution, NDSRollout], BaseSolution]

Repair = Callable[[BaseSolution, NDSRollout], BaseSolution]


def noop_improvement_step(
    solution: BaseSolution,
    model: NDSModel,
) -> BaseSolution:
    return solution


def apply_rollout(
    solution: CVRPSolution,
    rollout: NDSRollout,
) -> CVRPSolution:
    return solution.remove_customers(rollout.actions)


def only_expanding_apply_rollout(solution: BaseSolution, rollout: NDSRollout) -> BaseSolution:
    return solution.unsqueeze(1).expand(*solution.batch_size, rollout.log_p_of_actions.shape[1])


def noop_repair(
    solution: BaseSolution,
    rollout: NDSRollout,
) -> BaseSolution:
    return solution


def greedy_cvrp_repair(solution: CVRPSolution, rollout: NDSRollout) -> CVRPSolution:
    return GreedyCVRPRepair()(solution)
