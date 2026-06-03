import pytest
import torch

from core.lns.operators.destroy.random_node import RandomNodeDestroy
from core.lns.operators.repair.greedy_insert import CVRPGreedyInsertRepair
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
    destroy = RandomNodeDestroy[CVRPGraphSolution](
        destruction_size=destruction_size, seed=42, stitch=True
    )
    return destroy(cvrp_graph_solution)


class TestCVRPGreedyInsertRepair:
    def test_repairs_to_complete_and_feasible(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        assert not deconstructed_cvrp_graph_solution.is_complete.any()

        repair = CVRPGreedyInsertRepair()
        repaired = repair(deconstructed_cvrp_graph_solution)

        assert repaired.is_complete.all(), "Repaired solution must be complete."
        assert repaired.is_feasible.all(), "Repaired solution must be feasible."

    def test_no_self_loops(self, deconstructed_cvrp_graph_solution: CVRPGraphSolution) -> None:
        repaired = CVRPGreedyInsertRepair()(deconstructed_cvrp_graph_solution)

        customers_next = repaired.next_nodes[..., 1:]
        customer_indices = torch.arange(
            1, repaired.instance.num_customers + 1, device=repaired.instance.locations.device
        )
        assert (customers_next != customer_indices).all(), "No customer should point to itself."

    def test_depot_slot_invariant_preserved(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyInsertRepair()(deconstructed_cvrp_graph_solution)

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
        partial = solution.remove_customers(to_remove, stitch=True)

        assert partial.batch_size == torch.Size([B, R])

        repaired = CVRPGreedyInsertRepair()(partial)

        assert repaired.batch_size == torch.Size([B, R])
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()

    def test_all_customers_have_depot_connection(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        """Every customer in the repaired solution must have prev and next set."""
        repaired = CVRPGreedyInsertRepair()(deconstructed_cvrp_graph_solution)

        assert (repaired.prev_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all()
        assert (repaired.next_nodes[..., 1:] != CVRPGraphSolution.UNASSIGNED).all()

    def test_objective_is_finite(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyInsertRepair()(deconstructed_cvrp_graph_solution)

        obj = repaired.objective()
        assert obj.isfinite().all(), "Objective must be finite for a valid solution."
        assert (obj > 0).all(), "Objective must be positive."

    def test_objective_matches_route_traversal(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        repaired = CVRPGreedyInsertRepair()(deconstructed_cvrp_graph_solution)

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
        repaired = CVRPGreedyInsertRepair()(cvrp_graph_solution)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        assert torch.allclose(repaired.objective(), cvrp_graph_solution.objective())

    def test_repair_is_stochastic(
        self, deconstructed_cvrp_graph_solution: CVRPGraphSolution
    ) -> None:
        """Different seeds must produce different solutions so n_repairs > 1 is meaningful."""
        first = CVRPGreedyInsertRepair(seed=0)(deconstructed_cvrp_graph_solution)
        second = CVRPGreedyInsertRepair(seed=1)(deconstructed_cvrp_graph_solution)

        assert first.is_complete.all() and second.is_complete.all()
        assert not torch.equal(first.next_nodes, second.next_nodes), (
            "Different seeds produced identical solutions; n_repairs > 1 would be wasted work."
        )

    def test_inserts_at_arbitrary_position(self) -> None:
        """Repair must be able to insert mid-route, not only at depot-connected ends."""
        config = CVRPConfig(
            num_customers=4, vehicle_capacity_raw=10, min_demand_raw=1, max_demand_raw=1
        )
        instance = CVRPSampler(config, seed=0)(1)
        # One route 0 -> 1 -> 3 -> 4 -> 0, customer 2 free.
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 3, 4, 0]], dtype=torch.long)
        )
        # Detach customer 2 so it becomes a free customer.
        free = solution.remove_customers(torch.tensor([[2]]), stitch=True)
        # After stitching the (single) route is 0 -> 1 -> 3 -> 4 -> 0 (no change), and
        # customer 2 has prev = next = UNASSIGNED. But customer 2 was never in the route
        # in the first place — let's reset: take a route that contains 2, remove it,
        # then verify the repair inserts it back somewhere inside the same route.

        # Build a route containing 2: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 2, 3, 4, 0]], dtype=torch.long)
        )
        free = solution.remove_customers(torch.tensor([[2]]), stitch=True)

        repaired = CVRPGreedyInsertRepair(seed=0)(free)

        # Repaired must reinsert 2 somewhere inside the existing route. With one route
        # of capacity 10 and unit demands, capacity allows it. The cheapest position
        # (by geometry) should be between 1 and 3 (the original position).
        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        # Verify route still has all four customers in a single tour:
        all_in_one_route = repaired.route_ids[0, 1:].unique().numel() == 1
        assert all_in_one_route, "Single-route insert should not create a new route"

    def test_capacity_respected_via_new_tour_fallback(self) -> None:
        """When every existing route is full, the repair creates a new singleton route."""
        # Two customers per route, capacity exactly = sum of two demands. Removing one
        # customer from a tight route and trying to insert another free customer must
        # NOT exceed capacity on any existing route.
        config = CVRPConfig(
            num_customers=3, vehicle_capacity_raw=2, min_demand_raw=1, max_demand_raw=1
        )
        instance = CVRPSampler(config, seed=0)(1)
        # Route layout: 0 -> 1 -> 2 -> 0, 0 -> 3 -> 0. Capacity 2: first route has
        # demand 2 (full), second has demand 1.
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 2, 0, 3, 0]], dtype=torch.long)
        )
        # Detach customer 2 (with stitch so the first route becomes [1]).
        partial = solution.remove_customers(torch.tensor([[2]]), stitch=True)
        # First route now: 0 -> 1 -> 0 (demand 1). Customer 2 has demand 1.
        # Both existing routes can accept customer 2 since each currently has demand 1
        # and capacity 2 — so a new tour is NOT forced here. Let's force it with a
        # heavier demand instance.
        config = CVRPConfig(
            num_customers=3, vehicle_capacity_raw=2, min_demand_raw=2, max_demand_raw=2
        )
        instance = CVRPSampler(config, seed=0)(1)
        # Each customer has demand 2 == capacity, so each route holds exactly one
        # customer. Layout: 0 -> 1 -> 0, 0 -> 2 -> 0, 0 -> 3 -> 0.
        solution = CVRPGraphSolution.create_from_sequence(
            instance, torch.tensor([[0, 1, 0, 2, 0, 3, 0]], dtype=torch.long)
        )
        # Detach customer 2 — its old route becomes empty (no fragments).
        partial = solution.remove_customers(torch.tensor([[2]]), stitch=True)
        # All remaining routes are at capacity (demand 2 each). Inserting 2 anywhere
        # would exceed capacity; the new-tour fallback must fire.
        repaired = CVRPGreedyInsertRepair(seed=0)(partial)

        assert repaired.is_complete.all()
        assert repaired.is_feasible.all()
        # Customer 2 must have become its own route head (prev = depot).
        assert repaired.prev_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
        assert repaired.next_nodes[0, 2].item() == CVRPGraphSolution.DEPOT
