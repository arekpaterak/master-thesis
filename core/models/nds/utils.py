from core.models.nds.rollout import NDSRollout
from core.problems.cvrp import CVRPSolution


class RemoveCustomers:
    def __call__(self, solution: CVRPSolution, rollout: NDSRollout) -> CVRPSolution:
        return CVRPSolution(
            instance=solution.instance,
        )
