import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.greedy_reconnect import CVRPGreedyReconnectRepair
from core.problems.cvrp import CVRPConfig, CVRPSampler
from core.problems.cvrp.graph_solution import CVRPGraphSolution
from core.problems.cvrp.solve import one_customer_per_route_graph


@pytest.fixture(params=[5, 10])
def destruction_size(request) -> int:
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def cvrp_graph_solution(batch_size: int) -> CVRPGraphSolution:
    config = CVRPConfig.for_n(20)
    instance = CVRPSampler(config, seed=42)(batch_size)
    return one_customer_per_route_graph(instance)


@pytest.fixture
def deconstructed_cvrp_graph_solution(
    cvrp_graph_solution: CVRPGraphSolution, destruction_size: int
) -> CVRPGraphSolution:
    destroy = RandomNodeDestroy[CVRPGraphSolution](destruction_size=destruction_size, seed=42)
    return destroy(cvrp_graph_solution)


class TestCVRPGreedyReconnectRepair:
    def test_repairs_to_complete_and_feasible(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        assert not deconstructed_cvrp_graph_solution.is_complete.any()

        repair = CVRPGreedyReconnectRepair()
        repaired = repair(deconstructed_cvrp_graph_solution)

        assert repaired.is_complete.all(), "Repaired solution must be complete."
        assert repaired.is_feasible.all(), "Repaired solution must be feasible."

    def test_no_self_loops(self, deconstructed_cvrp_graph_solution: CVRPGraphSolution) -> None:
        repaired = CVRPGreedyReconnectRepair()(deconstructed_cvrp_graph_solution)

        customers_next = repaired.next_nodes[..., 1:]  # (..., N)
        customer_indices = torch.arange(
            1, repaired.instance.num_customers + 1, device=repaired.instance.locations.device
        )
        assert (customers_next != customer_indices).all(), "No customer should point to itself."

    def test_depot_slot_invariant_preserved(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyReconnectRepair()(deconstructed_cvrp_graph_solution)

        assert (repaired.next_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.prev_nodes[..., 0] == CVRPGraphSolution.UNASSIGNED).all()

    def test_rollout_batch_shape(self, batch_size: int) -> None:
        """Repair must work when solution has 2D batch shape (B, R) from remove_customers."""
        B, R, N, K = batch_size, 5, 20, 3
        config = CVRPConfig.for_n(N)
        instance = CVRPSampler(config, seed=42)(B)
        solution = one_customer_per_route_graph(instance)

        torch.manual_seed(42)
        to_remove = torch.randint(1, N + 1, (B, R, K))
        partial = solution.remove_customers(to_remove)

        assert partial.batch_size == torch.Size([B, R])

        repaired = CVRPGreedyReconnectRepair()(partial)

        assert repaired.batch_size == torch.Size([B, R])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_all_customers_have_depot_connection(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        """Every customer in the repaired solution must have prev and next set (no dangling ends)."""
        repaired = CVRPGreedyReconnectRepair()(deconstructed_cvrp_graph_solution)

        # Every customer must have a predecessor (prev != UNASSIGNED)
        assert (repaired.prev_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all(), (
            "All customers must have a predecessor after repair."
        )
        # Every customer must have a successor (next != UNASSIGNED)
        assert (repaired.next_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all(), (
            "All customers must have a successor after repair."
        )

    def test_objective_is_finite(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyReconnectRepair()(deconstructed_cvrp_graph_solution)

        obj = repaired.objective()
        assert obj.isfinite().all(), "Objective must be finite for a valid solution."
        assert (obj > 0).all(), "Objective must be positive."

    def test_objective_matches_route_traversal(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyReconnectRepair()(deconstructed_cvrp_graph_solution)

        B = repaired.batch_size[0]
        device = repaired.instance.locations.device
        reference = torch.zeros(B, device=device)

        for b in range(B):
            single = repaired[b]
            locs = single.instance.locations
            depot = torch.zeros(1, dtype=torch.long, device=device)
            for route in single.routes():
                nodes_in_route = torch.cat([depot, route.to(device), depot])
                route_locs = locs[nodes_in_route]
                reference[b] += (route_locs[1:] - route_locs[:-1]).pow(2).sum(-1).sqrt().sum()

        torch.testing.assert_close(repaired.objective(), reference)

    def test_already_complete_is_unchanged(self, cvrp_graph_solution: CVRPGraphSolution) -> None:
        """Calling repair on a complete solution should return it unchanged."""
        repaired = CVRPGreedyReconnectRepair()(cvrp_graph_solution)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        # Objective should match the original (no extra edges added)
        assert torch.allclose(repaired.objective(), cvrp_graph_solution.objective())
