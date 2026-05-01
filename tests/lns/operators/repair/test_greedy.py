#
import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.greedy import GreedyRepair
from core.problems.tsp import TSPConfig, TSPSampler, TSPSolution
from core.problems.tsp.solve import random_permutation


class TestGreedyRepair:
    @pytest.mark.parametrize(
        "batch_size, num_locations, destruction_size",
        [
            (1, 10, 3),
            (4, 20, 5),
        ],
    )
    def test_repairs_tsp(self, batch_size: int, num_locations: int, destruction_size: int) -> None:
        # Create full solution
        config = TSPConfig(num_locations=num_locations)
        instance = TSPSampler(config, seed=42)(batch_size)

        torch.manual_seed(42)
        solution = random_permutation(instance)

        assert solution.is_complete.all(), "Initial solution should be complete."

        # Destroy solution
        destroy = RandomNodeDestroy[TSPSolution](destruction_size=destruction_size, seed=42)
        destroyed_solution = destroy(solution)

        assert not destroyed_solution.is_complete.any(), (
            "Destroyed solution should not be complete."
        )

        # Repair solution
        repair = GreedyRepair[TSPSolution]()
        repaired_solution = repair(destroyed_solution)

        # Check if repaired successfully
        assert repaired_solution.is_complete.all(), "Repaired solution should be complete."
