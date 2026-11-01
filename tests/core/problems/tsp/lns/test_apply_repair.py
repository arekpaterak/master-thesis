import pytest
import torch

from core.lns.acceptance_criterion import GreedyAcceptance
from core.problems.tsp import TSPDistribution, TSPSampler
from core.problems.tsp.graph_solution import TSPGraphSolution
from core.problems.tsp.lns.apply_repair import TSPGreedyInsertApplyRepair
from core.problems.tsp.solve import random_permutation_graph
from tests.core.lns.operators.repair.apply_repair_templates import (
    ApplyRepairContract,
    DegreeTwoApplyRepairMixin,
)


class TestTSPApplyRepair(ApplyRepairContract, DegreeTwoApplyRepairMixin):
    @pytest.fixture
    def complete_solution(self, batch_size: int) -> TSPGraphSolution:
        instance = TSPSampler(TSPDistribution(num_locations=20), seed=42)(batch_size)
        return random_permutation_graph(instance)

    def make_apply_repair(self, **kwargs: object) -> TSPGreedyInsertApplyRepair:
        kwargs.setdefault("n_repairs", self.n_repairs)
        kwargs.setdefault("seed", self.repair_seed)
        return TSPGreedyInsertApplyRepair(**kwargs)

    def solution_state(self, solution: TSPGraphSolution) -> torch.Tensor:
        return solution.next_nodes

    # -- TSP specifics ----------------------------------------------------------

    def test_with_multidim_batch(self) -> None:
        # random_permutation_graph only supports a single leading batch dim, so build
        # the (identity) starting tour directly for this multi-dim-batch instance.
        instance = TSPSampler(TSPDistribution(num_locations=20), seed=1)(3).augment(2)
        sequence = torch.arange(instance.num_locations).expand(*instance.batch_size, -1)
        solution = TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)
        assert solution.batch_size == torch.Size([3, 2])

        selection = self.make_selection(solution, self.num_rollouts)
        repaired = self.make_apply_repair().sequential(solution, selection, GreedyAcceptance())

        assert repaired.batch_size == torch.Size([3, 2])
        assert repaired.is_complete.all()
