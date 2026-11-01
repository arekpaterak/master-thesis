from core.lns.acceptance_criterion import GreedyAcceptance
from core.lns.apply_and_repair import TorchApplyAndRepair
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.problems.tsp import TSPDistribution, TSPSampler
from core.problems.tsp.lns.operators.repair.insert import TSPGreedyInsertRepair
from core.problems.tsp.solve import random_permutation


def test_sequential_repairs_tsp_sequence_solution() -> None:
    instance = TSPSampler(TSPDistribution(num_locations=10), seed=0)(3)
    solution = random_permutation(instance, seed=0)
    selection = RandomNodeDestroy(destruction_size=4, seed=0).select(solution, num_rollouts=2)
    apply_and_repair = TorchApplyAndRepair(repair=TSPGreedyInsertRepair(seed=0), n_repairs=1)

    repaired = apply_and_repair.sequential(solution, selection, GreedyAcceptance())

    assert repaired.batch_size == solution.batch_size
    assert repaired.is_complete.all()
    assert (repaired.objective() <= solution.objective()).all()
