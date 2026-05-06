#
import pytest
import torch

from core.lns.operators.destroy.random_edge import TSPRandomEdgeDestroy
from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.greedy import (
    CVRPSequentialGreedyNodeInsertionRepair,
    GreedyNodeRepair,
)
from core.lns.operators.repair.greedy_edge import TSPGreedyEdgeRepair
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.solve import one_customer_per_route
from core.problems.tsp import TSPConfig, TSPSampler, TSPSolution
from core.problems.tsp.graph_solution import TSPGraphSolution
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
        repair = GreedyNodeRepair[TSPSolution]()
        repaired_solution = repair(destroyed_solution)

        # Check if repaired successfully
        assert repaired_solution.is_complete.all(), "Repaired solution should be complete."


class TestGreedyEdgeRepair:
    @pytest.mark.parametrize(
        "batch_size, num_locations, num_edges_to_destroy",
        [
            (1, 10, 3),
            (4, 20, 5),
        ],
    )
    def test_repairs_tsp(
        self, batch_size: int, num_locations: int, num_edges_to_destroy: int
    ) -> None:
        config = TSPConfig(num_locations=num_locations)
        instance = TSPSampler(config, seed=42)(batch_size)

        sequence = random_permutation(instance).tour
        solution = TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)

        assert solution.is_complete.all(), "Initial solution should be complete."

        # Destroy solution
        destroy = TSPRandomEdgeDestroy(destruction_size=num_edges_to_destroy, seed=42)
        destroyed_solution = destroy(solution)

        assert not destroyed_solution.is_complete.any(), (
            "Destroyed solution should not be complete."
        )

        # Repair solution
        repair = TSPGreedyEdgeRepair()
        repaired_solution = repair(destroyed_solution)

        # Check if repaired successfully
        assert repaired_solution.is_complete.all(), "Repaired solution should be complete."


class TestGreedyEdgeRepairWithNodeDestroy:
    @pytest.mark.parametrize(
        "batch_size, num_locations, destruction_size",
        [
            (1, 10, 3),
            (4, 20, 5),
            (4, 20, 2),  # adjacent removals likely in some batch elements
        ],
    )
    def test_repairs_tsp(self, batch_size: int, num_locations: int, destruction_size: int) -> None:
        config = TSPConfig(num_locations=num_locations)
        instance = TSPSampler(config, seed=42)(batch_size)

        sequence = random_permutation(instance).tour
        solution = TSPGraphSolution.create_from_sequence(instance=instance, sequence=sequence)

        assert solution.is_complete.all()

        destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)
        destroyed = destroy(solution)

        assert not destroyed.is_complete.any()
        # Each removed node severs 2 edges, so num_unassigned is between
        # destruction_size + 1 (all adjacent) and 2 * destruction_size (all non-adjacent).
        assert (destroyed.num_unassigned >= destruction_size + 1).all()
        assert (destroyed.num_unassigned <= 2 * destruction_size).all()

        repair = TSPGreedyEdgeRepair()
        repaired = repair(destroyed)

        assert repaired.is_complete.all()


class TestGreedyCVRPRepair:
    @pytest.mark.parametrize(
        "batch_size, num_customers, destruction_size",
        [
            (1, 20, 3),
            (4, 20, 5),
            (2, 50, 8),
        ],
    )
    def test_repairs_cvrp(self, batch_size: int, num_customers: int, destruction_size: int) -> None:
        config = CVRPConfig.for_n(num_customers)
        instance = CVRPSampler(config, seed=42)(batch_size)
        solution = one_customer_per_route(instance)

        assert solution.is_complete.all(), "Initial solution should be complete."
        assert solution.is_feasible.all(), "Initial solution should be feasible."

        destroy = RandomNodeDestroy(destruction_size=destruction_size, seed=42)
        destroyed = destroy(solution)

        assert not destroyed.is_complete.any(), "Destroyed solution should not be complete."

        repair = CVRPSequentialGreedyNodeInsertionRepair(seed=42)
        repaired = repair(destroyed)

        assert repaired.is_complete.all(), "Repaired solution should be complete."
        assert repaired.is_feasible.all(), "Repaired solution should be feasible."

    def test_repairs_with_rollout_batch_shape(self) -> None:
        """Verify repair works when solution has 2D batch shape (B, R) from remove_customers."""
        B, R, N, K = 2, 5, 20, 3
        config = CVRPConfig.for_n(N)
        instance = CVRPSampler(config, seed=42)(B)
        solution = one_customer_per_route(instance)

        to_remove = torch.randint(1, N + 1, (B, R, K))
        partial = solution.remove_customers(to_remove)

        assert partial.batch_size == torch.Size([B, R])

        repair = CVRPSequentialGreedyNodeInsertionRepair(seed=0)
        repaired = repair(partial)

        assert repaired.batch_size == torch.Size([B, R])
        assert repaired.is_complete.all(), "All rollout solutions should be complete."
        assert repaired.is_feasible.all(), "All rollout solutions should be feasible."
